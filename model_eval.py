import os
import torch
import numpy as np
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer as sa # SA Scoring from Ertl
from rdkit.Contrib.NP_Score import npscorer as natp # natural product likeliness
fscore = natp.readNPModel()
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from mflow.models.hyperparams import Hyperparameters as FlowHyperPars
from mflow.models.model import MoFlow, rescale_adj
from mflow.utils.molecular_metrics import *
from mflow.models.utils import adj_to_smiles, construct_mol, construct_mol_with_validation
from mflow.generate import generate_mols
from data import transform_qm9
from data.data_loader import NumpyTupleDataset

# this use of qm9-5k may not be correct, maybe use the full qm9?
from data.sparse_qm9.sparse_molecular_dataset import SparseMolecularDataset
train_sparse= SparseMolecularDataset()
# train_sparse.load('./data/sparse_qm9/qm9_5k.sparsedataset')
train_sparse.load('./data/sparse_qm9/qm9.sparsedataset')


def score_model(path, num_expt=1, return_properties=False, batch_size=1000):
    '''
    Takes the path to a pre-trained model checkpoint, generates 1000 molecules, and scores them

    Returns:
    nuvd: [novelty, uniqueness, validity, diversity]
    avg_scores: avg[np likeness, logP, SA, QED, drug candidacy]
    properties: per molecule properties (if return_properties=True)
    '''
    # load pre-trained model from checkpoint
    gen = load_model(path)

    nuvds = np.zeros((1,4))
    avgs = np.zeros((1,5))

    for i in range(num_expt):
        adj, x = generate_mols(gen, batch_size=batch_size)
        nuvd, properties = evaluate_scores(adj, x)
        avg_scores = np.mean(properties, axis=0).flatten()
        nuvds = np.vstack((nuvds, nuvd))
        avgs = np.vstack((avgs, avg_scores))
    
    nuvds = nuvds[1:, :].mean(axis=0).tolist()
    avgs = avgs[1:, :].mean(axis=0).tolist()

    if return_properties: return nuvds, avgs, properties
    else: return nuvds, avgs


def score_reconstruction(path, gpu=-1):
    '''

    '''
    # load pre-trained model from checkpoint
    gen = load_model(path)

    if gpu >= 0:
        device = torch.device('cuda:'+str(gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    gen.to(device)

    # prep dataset
    atomic_num_list = [6, 7, 8, 9, 0]
    transform_fn = transform_qm9.transform_fn
    valid_idx = transform_qm9.get_val_ids()
    molecule_file = 'qm9_relgcn_kekulized_ggnp.npz'
    dataset = NumpyTupleDataset.load(os.path.join('./data', molecule_file), transform=transform_fn)

    batch_size = 256
    assert len(valid_idx) > 0
    train_idx = [t for t in range(len(dataset)) if t not in valid_idx]  # 120803 = 133885-13082
    train = torch.utils.data.Subset(dataset, train_idx)  # 120803
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size)

    # evaluate dataset reconstruction
    reconstruction_rate_list = []
    for i, batch in enumerate(train_dataloader):
        x = batch[0].to(device)  # (256, 9, 5)
        adj = batch[1].to(device)  # (256, 4, 9, 9)
        adj_normalized = rescale_adj(adj).to(device)
        z, _ = gen(adj, x, adj_normalized)
        z0 = z[0].reshape(z[0].shape[0], -1)
        z1 = z[1].reshape(z[1].shape[0], -1)
        adj_rev, x_rev = gen.reverse(torch.cat([z0, z1], dim=1))
        reverse_smiles = adj_to_smiles(adj_rev.cpu(), x_rev.cpu(), atomic_num_list)
        train_smiles = adj_to_smiles(adj.cpu(), x.cpu(), atomic_num_list)
        lb = np.array([int(a != b) for a, b in zip(train_smiles, reverse_smiles)])
        idx = np.where(lb)[0]
        if len(idx) > 0:
            for k in idx:
                print(i*batch_size+k, 'train: ', train_smiles[k], ' reverse: ', reverse_smiles[k])
        reconstruction_rate = 1.0 - lb.mean()
        reconstruction_rate_list.append(reconstruction_rate)
    reconstruction_rate_total = np.array(reconstruction_rate_list).mean()
    print("reconstruction_rate for all the train data:{} in {}".format(reconstruction_rate_total, len(train)))
    return reconstruction_rate_total


## auxiliary scoring functions
def load_model(path):
    '''
    Loads a model from the checkpoint file
    '''
    param_path = os.path.join(os.path.dirname(path), 'gen-params.json')
    model_params_gflow = FlowHyperPars(path=param_path)
    gen = MoFlow(model_params_gflow)
    chk = torch.load(path, map_location='cpu')
    gen.load_state_dict(chk['GStateDict'])
    return gen


def true_synthetic_accessibility_scores(sanitized_mols):
    '''
    Synthetic accessability score: [1, 10] - lower is better
    As described in http://www.jcheminf.com/content/1/1/8
    '''
    scores = np.array([sa.calculateScore(mol) if mol is not None 
                       else 10 for mol in sanitized_mols])  # [1, 10]
    return scores


def true_natural_product_scores(sanitized_mols):
    '''
    Ertl's natural product likness scores
    Originally: [-5, 5] - higher is better
    
    http://pubs.acs.org/doi/abs/10.1021/ci700286x
    '''
    scores = np.array([natp.scoreMol(mol, fscore) if mol is not None 
                       else -5 for mol in sanitized_mols])  # [-5, 5]
    return scores


def drug_candidate_scores(logP, syn_acc, nov):
    scores = (constant_bump(logP, 0.210, 0.945) + syn_acc + nov + (1 - nov) * 0.3) / 4
    return scores # open ended


def constant_bump(x, x_low, x_high, decay=0.025):
    return np.select(condlist=[x <= x_low, x >= x_high],
                        choicelist=[np.exp(- (x - x_low) ** 2 / decay),
                                    np.exp(- (x - x_high) ** 2 / decay)],
                        default=np.ones_like(x))


def evaluate_scores(edges, nodes, atomic_num_list=[6, 7, 8, 9, 0], training_data=train_sparse):
    def to_mol(adj_x):
        adj_elem, x_elem = adj_x
        return construct_mol(x_elem, adj_elem, atomic_num_list)
    
    def to_validated_mol(adj_x):
        adj_elem, x_elem = adj_x
        return construct_mol_with_validation(x_elem, adj_elem, atomic_num_list)
    
    def clean_sanitize(mol):
        '''
        There's an issue with trying to sanitize fake molecules.
        Sanitization doesn't remove/mark invalid so we use flags as a mask
        0 - invalid
        1 - valid and sanitized
        
        Solution adapted from: https://github.com/rdkit/rdkit/issues/2216
        '''
        try:
            Chem.SanitizeMol(mol)
            return 1
        except:
            return 0

    adj = edges.__array__()  # (bs,4,9,9)
    x = nodes.__array__()  # (bs,9,5)
            
    mols = list(map(to_mol, zip(adj, x)))
    valid_mols = list(map(to_validated_mol, zip(adj, x)))
    flags = list(map(clean_sanitize, valid_mols))  # see clean_sanitize spec
    sani_mols = [mol if validity == 1 else None for mol, validity in zip(valid_mols, flags)]
    
    mm = MolecularMetrics()
    novelty = mm.novel_scores(mols, training_data).mean()  # scalar
    uniqueness = mm.unique_scores(mols).mean()   # scalar
    validity = mm.valid_scores(mols).mean()   # scalar
    diversity = mm.diversity_scores(sani_mols, training_data).mean() # scalar
    
    water_octanol_partition = mm.water_octanol_partition_coefficient_scores(mols, norm=False).flatten().reshape(-1,1) # vec - [0, 1]
    qed = mm.quantitative_estimation_druglikeness_scores(mols, norm=False).flatten().reshape(-1,1)  # vec - [0, 1]
    np_score = true_natural_product_scores(sani_mols).reshape(-1,1) # vec - [1, 10]
    synthetic_accessibility = true_synthetic_accessibility_scores(sani_mols).reshape(-1,1)  # vec - [-5, 5]
    drug_candidacy = drug_candidate_scores(water_octanol_partition, 
        synthetic_accessibility, novelty).reshape(-1,1)  # vec - [0,1]

    scores = np.hstack((np_score, water_octanol_partition, synthetic_accessibility, qed, drug_candidacy))
    nuvd = np.array([novelty, uniqueness, validity, diversity])

    return nuvd, scores

