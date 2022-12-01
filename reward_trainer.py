import json
import os
import sys
# for linux env.
# sys.path.insert(0,'.')
import argparse
from distutils.util import strtobool
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.data_loader import NumpyTupleDataset

from mflow.models.hyperparams import Hyperparameters as FlowHyperPars
from mflow.models.model import MoFlow, rescale_adj
from mflow.models.utils import check_validity, save_mol_png
from mGAN.hyperparams import Hyperparameters as DiscHyperPars
from mGAN.models import Discriminator
from mGAN.reward_loss import calculate_rewards

import time
from mflow.utils.timereport import TimeReport
from mflow.generate import generate_mols

import functools
print = functools.partial(print, flush=True)

 
def get_parser():
    parser = argparse.ArgumentParser()
    # data I/O
    parser.add_argument('-i', '--data_dir', type=str, default='./data', help='Location for the dataset')
    parser.add_argument('--data_name', type=str, default='qm9', choices=['qm9', 'zinc250k'], help='dataset name')
    # parser.add_argument('-f', '--data_file', type=str, default='qm9_relgcn_kekulized_ggnp.npz', help='Name of the dataset')
    parser.add_argument('-o', '--save_dir', type=str, default='results/qm9',
                        help='Location for parameter checkpoints and samples')
    parser.add_argument('-t', '--save_interval', type=int, default=20,
                        help='Every how many epochs to write checkpoint/samples?')
    parser.add_argument('-r', '--load_params', type=int, default=0,
                        help='Restore training from previous model checkpoint? 1 = Yes, 0 = No')
    parser.add_argument('--load_snapshot', type=str, default='', help='load the model from this path')

    # optimization
    parser.add_argument('--adv_reg', type=float, default=0.1, help='GAN loss tradeoff')
    parser.add_argument('--rl_reg', type=float, default=0.1, help='Reward learning loss tradeoff')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
    parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('-b1', '--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer')
    parser.add_argument('-b2', '--beta2', type=float, default=0.999, help='Beta2 for Adam optimizer')
    parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU Id to use')
    parser.add_argument('--save_epochs', type=int, default=1, help='in how many epochs, a snapshot of the model'
                                                                   ' needs to be saved?')
    # data loader
    parser.add_argument('-b', '--batch_size', type=int, default=256, help='Batch size during training per GPU')
    parser.add_argument('--shuffle', type=strtobool, default='true', help='Shuffle the data batch')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers in the data loader')

    # # evaluation
    # parser.add_argument('--sample_batch_size', type=int, default=16,
    #                     help='How many samples to process in paralell during sampling?')
    # reproducibility

    # For bonds
    parser.add_argument('--b_n_flow', type=int, default=10,
                        help='Number of masked glow coupling layers per block for bond tensor')
    parser.add_argument('--b_n_block', type=int, default=1, help='Number of glow blocks for bond tensor')
    parser.add_argument('--b_hidden_ch', type=str, default="128,128",
                        help='Hidden channel list for bonds tensor, delimited list input ')
    parser.add_argument('--b_conv_lu', type=int, default=1, choices=[0, 1, 2],
                        help='0: InvConv2d for 1*1 conv, 1:InvConv2dLU for 1*1 conv, 2: No 1*1 conv, '
                             'swap updating in the coupling layer')
    # For atoms
    parser.add_argument('--a_n_flow', type=int, default=27,
                        help='Number of masked flow coupling layers per block for atom matrix')
    parser.add_argument('--a_n_block', type=int, default=1, help='Number of flow blocks for atom matrix')
    parser.add_argument('--a_hidden_gnn', type=str, default="64,",
                        help='Hidden dimension list for graph convolution for atoms matrix, delimited list input ')
    parser.add_argument('--a_hidden_lin', type=str, default="128,64",
                        help='Hidden dimension list for linear transformation for atoms, delimited list input ')
    parser.add_argument('--mask_row_size_list', type=str, default="1,",
                        help='Mask row size list for atom matrix, delimited list input ')
    parser.add_argument('--mask_row_stride_list', type=str, default="1,",
                        help='Mask row stride list for atom matrix, delimited list input')

    # Discriminator network
    parser.add_argument('--disc_conv_dim', type=list, default=[[128, 64],128, [128, 64]],
                        help='Discriminator convolution dimensions (graph_conv_dim, aux_dim, linear_dim)')
    parser.add_argument('--disc_with_features', type=bool, default=False,
                         help='')
    parser.add_argument('--disc_f_dim', type=int, default=0,
                         help='')
    parser.add_argument('--disc_dropout_rate',  type=float, default=0.0,
                         help='')
    parser.add_argument('--disc_activation', type=str, default='tanh',
                         help='')
    parser.add_argument('--disc_lam', type=float, default=10.0,
                         help='')

    # General
    parser.add_argument('-s', '--seed', type=int, default=420, help='Random seed to use')
    parser.add_argument('--debug', type=strtobool, default='true', help='To run training with more information')
    parser.add_argument('--learn_dist', type=strtobool, default='true', help='learn the distribution of feature matrix')
    parser.add_argument('--noise_scale', type=float, default=0.6, help='x + torch.rand(x.shape) * noise_scale')

    return parser


def gradient_penalty(y, x, device):
    '''Compute gradient penalty: (L2_norm(dy/dx) - 1)**2.'''
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                                inputs=x,
                                grad_outputs=weight,
                                retain_graph=True,
                                create_graph=True,
                                only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
    return torch.mean((dydx_l2norm - 1) ** 2)


def make_noise(batch_size, a_n_node, a_n_type, b_n_type, device):
    '''
    Generate a random noise tensor
    In: B = Batch size, N = number of atoms (a_n_node), M = number of bond types (b_n_types), 
        T = number of atom types (Carbon, Oxygen etc.) (a_n_type)
    Out: z: latent vector. Shape: [B, N*N*M + N*T] 
    '''
    return torch.randn(batch_size, a_n_node * a_n_node * b_n_type + a_n_node * a_n_type, device=device, requires_grad=True)


def postprocess(inputs, method, temperature=1.):
    def listify(x):
        return x if type(x) == list or type(x) == tuple else [x]

    def delistify(x):
        return x if len(x) > 1 else x[0]

    if method == 'soft_gumbel':
        softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                                    / temperature, hard=False).view(e_logits.size())
                    for e_logits in listify(inputs)]
    elif method == 'hard_gumbel':
        softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                                    / temperature, hard=True).view(e_logits.size())
                    for e_logits in listify(inputs)]
    else:
        softmax = [F.softmax(e_logits / temperature, -1)
                    for e_logits in listify(inputs)]

    return [delistify(e) for e in (softmax)]


def train():
    start = time.time()
    print("Start at Time: {}".format(time.ctime()))
    parser = get_parser()
    args = parser.parse_args()

    # use GPUs if available
    device = -1
    multigpu = False
    if args.gpu == -1:
        # cpu
        device = torch.device('cpu')
    elif args.gpu >= 0:
        # single gpu
        device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    else:
        # multigpu, can be slower than using just 1 gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        multigpu = True

    debug = args.debug

    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args

    # Flow model configuration
    b_hidden_ch = [int(d) for d in args.b_hidden_ch.strip(',').split(',')]
    a_hidden_gnn = [int(d) for d in args.a_hidden_gnn.strip(',').split(',')]
    a_hidden_lin = [int(d) for d in args.a_hidden_lin.strip(',').split(',')]
    mask_row_size_list = [int(d) for d in args.mask_row_size_list.strip(',').split(',')]
    mask_row_stride_list = [int(d) for d in args.mask_row_stride_list.strip(',').split(',')]
    if args.data_name == 'qm9':
        from data import transform_qm9
        data_file = 'qm9_relgcn_kekulized_ggnp.npz'
        transform_fn = transform_qm9.transform_fn
        atomic_num_list = [6, 7, 8, 9, 0]
        b_n_type = 4
        b_n_squeeze = 3
        a_n_node = 9
        a_n_type = len(atomic_num_list)  # 5
        valid_idx = transform_qm9.get_val_ids()  # len: 13,082, total data: 133,885
    elif args.data_name == 'zinc250k':
        from data import transform_zinc250k
        data_file = 'zinc250k_relgcn_kekulized_ggnp.npz'
        transform_fn = transform_zinc250k.transform_fn_zinc250k
        atomic_num_list = transform_zinc250k.zinc250_atomic_num_list  # [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
        # mlp_channels = [1024, 512]
        # gnn_channels = {'gcn': [16, 128], 'hidden': [256, 64]}
        b_n_type = 4
        b_n_squeeze = 19   # 2
        a_n_node = 38
        a_n_type = len(atomic_num_list)  # 10
        valid_idx = transform_zinc250k.get_val_ids()
    else:
        raise ValueError('Only support qm9 and zinc250k right now. '
                         'Parameters need change a little bit for other dataset.')

    ## Make generator model
    model_params_gflow = FlowHyperPars(b_n_type=b_n_type,  # 4,
                                   b_n_flow=args.b_n_flow,
                                   b_n_block=args.b_n_block,
                                   b_n_squeeze=b_n_squeeze,
                                   b_hidden_ch=b_hidden_ch,
                                   b_affine=True,
                                   b_conv_lu=args.b_conv_lu,
                                   a_n_node=a_n_node,
                                   a_n_type=a_n_type,
                                   a_hidden_gnn=a_hidden_gnn,
                                   a_hidden_lin=a_hidden_lin,
                                   a_n_flow=args.a_n_flow,
                                   a_n_block=args.a_n_block,
                                   mask_row_size_list=mask_row_size_list,
                                   mask_row_stride_list=mask_row_stride_list,
                                   a_affine=True,
                                   learn_dist=args.learn_dist,
                                   seed=args.seed,
                                   noise_scale=args.noise_scale
                                   )
    print('Generator params:')
    model_params_gflow.print()
    gen = MoFlow(model_params_gflow)
    os.makedirs(args.save_dir, exist_ok=True)
    gen.save_hyperparams(os.path.join(args.save_dir, 'gen-params.json'))
    if torch.cuda.device_count() > 1 and multigpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        gen = nn.DataParallel(gen)
    else:
        multigpu = False
    gen = gen.to(device)

    ## Make discriminator model
    model_params_disc = DiscHyperPars(b_n_type= b_n_type,  # 4
                                      a_n_node= a_n_node,  # 9
                                      a_n_type= a_n_type,  # 5
                                      conv_dim= args.disc_conv_dim,  # [[128, 64], 128, [128, 64]]
                                      with_features= args.disc_with_features,  # False
                                      f_dim= args.disc_f_dim,  # 0
                                      lam= args.disc_lam,  # 10
                                      dropout_rate= args.disc_dropout_rate,  # 0.
                                      activation= args.disc_activation,  # tanh
                                      seed= args.seed
                                      )
    print('Discriminator params:')
    model_params_disc.print()
    disc = Discriminator(model_params_disc)
    os.makedirs(args.save_dir, exist_ok=True)
    disc.save_hyperparams(os.path.join(args.save_dir, 'disc-params.json'))
    if torch.cuda.device_count() > 1 and multigpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        disc = nn.DataParallel(disc)
    else:
        multigpu = False
    disc = disc.to(device)
    
    ## Make reward model
    print('Reward network params:')
    model_params_disc.print()
    rew = Discriminator(model_params_disc) # use same hyperparams as discriminator
    os.makedirs(args.save_dir, exist_ok=True)
    rew.save_hyperparams(os.path.join(args.save_dir, 'rew-params.json'))
    if torch.cuda.device_count() > 1 and multigpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        rew = nn.DataParallel(rew)
    else:
        multigpu = False
    rew = rew.to(device)


    # auxiliary disciminator patch function
    def label_2_onehot(labels, dim):
        '''Convert label indices to one-hot vectors.'''
        out = torch.zeros(list(labels.size())).to(device)
        out.scatter_(len(out.size()) - 1, labels.type(torch.int64), 1.)
        return out

    # Datasets:
    dataset = NumpyTupleDataset.load(os.path.join(args.data_dir, data_file), transform=transform_fn)  # 133885
    if len(valid_idx) > 0:
        train_idx = [t for t in range(len(dataset)) if t not in valid_idx]  # 120803 = 133885-13082
        # n_train = len(train_idx)  # 120803
        train = torch.utils.data.Subset(dataset, train_idx)  # 120,803
        test = torch.utils.data.Subset(dataset, valid_idx)  # 13,082
    else:
        torch.manual_seed(args.seed)
        train, test = torch.utils.data.random_split(
            dataset,
            [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size,
                                                   shuffle=args.shuffle, num_workers=args.num_workers)

    print('==========================================')
    print('Load data done! Time {:.2f} seconds'.format(time.time() - start))
    print('Data shuffle: {}, Number of data loader workers: {}!'.format(args.shuffle, args.num_workers))
    print('Device: {}!'.format(device))
    if args.gpu >= 0:
        print('Using GPU device:{}!'.format(args.gpu))
    print('Num Train-size: {}'.format(len(train)))
    print('Num Minibatch-size: {}'.format(args.batch_size))
    print('Num Iter/Epoch: {}'.format(len(train_dataloader)))
    print('Num epoch: {}'.format(args.max_epochs))
    print('Adversarial loss coefficient: {}'.format(args.adv_reg))
    print('Reward loss coefficient: {}'.format(args.rl_reg))
    print('==========================================')


    # loss and optimizers
    optimizer_gen = torch.optim.Adam(gen.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
    optimizer_disc = torch.optim.Adam(disc.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
    optimizer_rew = torch.optim.Adam(rew.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
    adv = args.adv_reg
    rl = args.rl_reg
    

    # keep track of training
    disc_losses = []
    rew_losses = [0]
    gen_losses = [0]

    # Train the models
    iter_per_epoch = len(train_dataloader)
    gen_iter = 0
    log_step = args.save_interval  # 20 default
    tr = TimeReport(total_iter=args.max_epochs * iter_per_epoch)
    for epoch in range(args.max_epochs):
        print("In epoch {}, Time: {}".format(epoch+1, time.ctime()))
        for i, batch in enumerate(train_dataloader):
            x = batch[0].to(device)  # (256, 9, 5)
            adj = batch[1].to(device)  # (256, 4, 9, 9)
            adj_normalized = rescale_adj(adj).to(device)
            x_onehot = label_2_onehot(x, a_n_type).to(device)
            adj_onehot = label_2_onehot(adj, b_n_type).to(device)

            # two time-scale training
            if gen_iter < 25 or gen_iter % 500 == 0:
                train_gen = True if i % 100 == 0 else False  #### TODO: Check this is 100
            else:
                train_gen = True if i % 8 == 0 else False
            
            # ==============================================================
            #         Discriminator training step
            # The generator is trained using the Wasserstein-GAN + gradient penalty
            # objective described by Gulrajani et al. (https://arxiv.org/abs/1704.00028)
            # ==============================================================
            # zero gradients
            gen.zero_grad()
            disc.zero_grad()
            rew.zero_grad()                         

            # real batch 
            logits_real, _ = disc(adj_onehot, None, x_onehot)

            # fake batch
            # reverse pass through generator
            edges, nodes = gen.reverse(make_noise(x.size()[0], a_n_node, a_n_type, b_n_type, device))
            # gumbel softmax
            e_hat, n_hat = postprocess((edges, nodes), 'medium_gumbel')
            # get fake batch logits
            logits_fake, _ = disc(e_hat, None, n_hat)

            # compute gradient penalty
            eps = torch.rand(logits_real.size(0), 1, 1, 1).to(device)
            x_int0 = (eps * adj_onehot + (1. - eps) * e_hat).requires_grad_(True)
            x_int1 = (eps.squeeze(-1) * x_onehot + (1. - eps.squeeze(-1)) * n_hat).requires_grad_(True)
            grad0, grad1 = disc(x_int0, None, x_int1)
            grad_penalty = gradient_penalty(grad0, x_int0, device) + gradient_penalty(grad1, x_int1, device)

            # compute wGAN losses + objective
            disc_loss_real = torch.mean(logits_real)
            disc_loss_fake = torch.mean(logits_fake)
            disc_loss = -disc_loss_real + disc_loss_fake + disc.lam * grad_penalty
            disc_loss.backward()  # backwards pass
            optimizer_disc.step()  # update discriminator
            
            disc_losses.append(disc_loss.item())
            rew_losses.append(rew_losses[-1])
            gen_losses.append(gen_losses[-1])                       
            
            
            # ==============================================================
            #         Generator training step
            # The generator is trained using the hybrid objective described
            # by Ermon et al. (https://arxiv.org/abs/1705.08868)
            # See below for more details
            # ==============================================================
            if train_gen or i == len(train_dataloader) - 1:
                gen.zero_grad()
                disc.zero_grad()
                rew.zero_grad()


                ## likelihood training step
                # forward pass through flow generator
                z, sum_log_det_jacs = gen(adj, x, adj_normalized)
                # calculate nll
                if multigpu:
                    nll = gen.module.log_prob(z, sum_log_det_jacs)
                else:
                    nll = gen.log_prob(z, sum_log_det_jacs)
                nll_loss = nll[0] + nll[1]


                ## adversarial training step
                # generate a fake batch
                edges, nodes = gen.reverse(make_noise(x.size()[0], a_n_node, a_n_type, b_n_type, device))
                # gumbel softmax
                e_hat, n_hat = postprocess((edges, nodes), 'medium_gumbel')
                # get fake batch logits
                logits_fake, _ = disc(e_hat, None, n_hat)  # calculate GAN loss | log(D(G(z)))
                gan_loss = -logits_fake.mean()    # -log(D(G(z)))
                

                ## reward loss training step                    
                # real batch rewards
                pred_rewards_real, _ = rew(adj_onehot, None, x_onehot, activation = nn.Sigmoid())
                rewards_real = calculate_rewards(adj, x, atomic_num_list)
                rewards_real = torch.from_numpy(rewards_real[:,np.newaxis]).to(device, dtype=torch.float32)
            
                # fake batch rewards
                pred_rewards_fake, _ = rew(e_hat, None, n_hat, activation = nn.Sigmoid())
                (e_hard, n_hard) = postprocess((e_hat, n_hat), 'hard_gumbel')
                # e_hard, n_hard = torch.max(e_hard, -1), torch.max(n_hard, -1)
                rewards_fake = calculate_rewards(e_hard, n_hard, atomic_num_list)
                rewards_fake = torch.from_numpy(rewards_fake[:,np.newaxis]).to(device, dtype=torch.float32)
                rl_loss = -pred_rewards_fake.mean()

                '''
                MoFlowGAN generator loss formulation
                D: discriminator network
                G: normalizing flow generator
                R: reward network
                z: latent vector
                c: RL/Adv tradeoff parameter
                a: molGAN alpha = loss_gan/loss_rl

                Flow: max [ll] == min [nll]
                GAN: max [log(D(G(z)))] == min [-log(D(G(z)))]
                molGAN w/ RL: min [c * -log(D(G(z))) + (1 - c) * a * R(G(z))]
                FlowGAN: min[nll + -log(D(G(z))) + R(G(z))]

                Our implementation: min (1-c) * nll + adv * -log(D(G(z))) + rl * a * fake_rew_logits
                '''
                # compute generator losses
                if epoch + 1 <= 4: 
                    rl = 0       # for the first few epochs don't use the RL objective to train the generator
                    alpha = 0    # see molGAN paper for full explanation - https://arxiv.org/abs/1805.11973
                else:
                    with torch.no_grad():
                        alpha = torch.abs(gan_loss / rl_loss)
                
                gen_loss = (1 - adv - rl) * nll_loss + (adv * gan_loss) + (rl * alpha * rl_loss)  # calculate weighted total loss
                gen_loss.backward(retain_graph=True)  # backwards pass through generator
                optimizer_gen.step()  # update generator


                # compute reward network loss (MSE) - loss_V from molGAN
                rew_loss = (pred_rewards_fake - rewards_fake) ** 2 + (pred_rewards_real - rewards_real) ** 2
                rew_loss = rew_loss.mean()
                rew_loss.backward()  # backwards pass through reward network
                optimizer_rew.step()  # update reward network

                # keep track of network losses
                disc_losses.append(disc_losses[-1])
                rew_losses.append(rew_loss.item())
                gen_losses.append(gen_loss.item())
                gen_iter+= 1
            

            tr.update()

            # Print log info
            if (i+1) % log_step == 0:  # i % args.log_step == 0:
                print('Epoch [{}/{}], Iter [{}/{}], gen_loss: {:.5f}, disc_loss: {:.5f} '
                      'disc_loss_reals: {:.3f}, disc_loss_fakes: {:.3f}, rew_loss: {:.5f} '
                      '{:.2f} sec/iter, {:.2f} iters/sec'.
                      format(epoch+1, args.max_epochs, i+1, iter_per_epoch, gen_loss.item(),
                             disc_losses[-1], disc_loss_real, disc_loss_fake, rew_losses[-1],
                             tr.get_avg_time_per_iter(), tr.get_avg_iter_per_sec()))
                tr.print_summary()

        if debug:
            def print_validity(ith):
                gen.eval()
                if multigpu:
                    adj, x = generate_mols(gen.module, batch_size=100, device=device)
                else:
                    adj, x = generate_mols(gen, batch_size=100, device=device)
                valid_mols = check_validity(adj, x, atomic_num_list)['valid_mols']
                # mol_dir = os.path.join(args.save_dir, 'generated_{}'.format(ith))
                # os.makedirs(mol_dir, exist_ok=True)
                # for ind, mol in enumerate(valid_mols):
                #     save_mol_png(mol, os.path.join(mol_dir, '{}.png'.format(ind)))
                gen.train()
            print_validity(epoch+1)

        # The same report for each epoch
            print('Epoch [{}/{}], Iter [{}/{}], gen_loss: {:.5f}, nll_x: {:.5f}, '
                    'nll_adj: {:.5f}, adv: {:.3f}, gan_loss: {:.5f}, disc_loss: {:.5f}, '
                    'gen_training_iters: {}, {:.2f} sec/iter, {:.2f} iters/sec'.
                    format(epoch+1, args.max_epochs, i+1, iter_per_epoch, gen_loss.item(),
                            nll[0].item(), nll[1].item(), adv, gan_loss.item(), disc_losses[-1],
                            gen_iter, tr.get_avg_time_per_iter(), tr.get_avg_iter_per_sec()))
        tr.print_summary()

        # Save the model checkpoints
        save_epochs = args.save_epochs
        if save_epochs == -1:
            save_epochs = args.max_epochs
        if ((epoch + 1) % save_epochs == 0) or (epoch + 1 == args.max_epochs):
            if multigpu:
                torch.save(gen.module.state_dict(), os.path.join(
                args.save_dir, 'model_snapshot_epoch_{}'.format(epoch + 1)))
            else:
                saveModel(gen, gen_losses, disc_losses, rew_losses, gen_iter, os.path.join(
                args.save_dir, 'model_snapshot_epoch_{}'.format(epoch + 1)))
            tr.end()

    print("[Training Ends], Start at {}, End at {}".format(time.ctime(start), time.ctime()))

def saveModel(G, gen_losses, disc_losses, rew_losses, gen_iters, path):
    torch.save({
        'GStateDict': G.state_dict(),
        'gLosses': gen_losses,
        'dLosses': disc_losses,
        'rLosses': rew_losses,
        'genIters': gen_iters,
        }, path)

if __name__ == '__main__':
    # with torch.autograd.set_detect_anomaly(True):
    train()
