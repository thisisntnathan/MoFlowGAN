import os
import torch
from rdkit.Chem import Draw

from mflow.models.hyperparams import Hyperparameters as FlowHyperPars
from mflow.models.model import MoFlow
from mflow.models.utils import check_validity
from mflow.generate import generate_mols

def show_and_tell(path):
    '''
    Generate 100 molecules using the model and draw them
    '''
    model_params_gflow = FlowHyperPars(b_n_type=4,
                            b_n_flow=10,
                            b_n_block=1,
                            b_n_squeeze=3,
                            b_hidden_ch=[128, 128],
                            b_affine=True,
                            b_conv_lu=1,
                            a_n_node=9,
                            a_n_type=5,
                            a_hidden_gnn=[64],
                            a_hidden_lin=[128, 64],
                            a_n_flow=27,
                            a_n_block=1,
                            mask_row_size_list=[1],
                            mask_row_stride_list=[1],
                            a_affine=True,
                            learn_dist=1,
                            seed=420,
                            noise_scale=0.6)
    gen = MoFlow(model_params_gflow)
    chk = torch.load(path)
    gen.load_state_dict(chk['GStateDict'])
    atomic_num_list=[6, 7, 8, 9, 0]

    adj, x = generate_mols(gen, batch_size=100)
    val_res = check_validity(adj, x, atomic_num_list)
    
    pth = os.path.normpath(os.path.dirname(path))
    gen_dir = '/notebooks/results/validation/generated'
    os.makedirs(gen_dir, exist_ok=True)
    filepath = os.path.join(gen_dir, 'generated_mols_{}.png'.format(pth[-1]))
    img = Draw.MolsToGridImage(val_res['valid_mols'], legends=val_res['valid_smiles'], 
                               molsPerRow=10, subImgSize=(300, 300), returnPNG=False)
    img.save(filepath)
