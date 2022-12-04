import time
import torch
import numpy as np
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer as sa # SA Scoring from Ertl
from rdkit.Contrib.NP_Score import npscorer as natp # natural product likeliness
fscore = natp.readNPModel()
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from mflow.models.hyperparams import Hyperparameters as FlowHyperPars
from mflow.models.model import MoFlow
from mflow.utils.molecular_metrics import *
from mflow.models.utils import construct_mol, construct_mol_with_validation
from mflow.generate import generate_mols
# from mGAN.reward_loss import synthetic_accessibility_scores, natural_product_scores

# this use of qm9-5k may not be correct, maybe use the full qm9?
from data.sparse_molecular_dataset import SparseMolecularDataset
train_sparse= SparseMolecularDataset()
train_sparse.load('./data/qm9_5k.sparsedataset')


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
    
    water_octanol_partition = mm.water_octanol_partition_coefficient_scores(mols, norm=True).flatten().reshape(-1,1) # vec - [0, 1]
    qed = mm.quantitative_estimation_druglikeness_scores(mols, norm=True).flatten().reshape(-1,1)  # vec - [0, 1]
    np_score = true_natural_product_scores(sani_mols).reshape(-1,1) # vec - [1, 10]
    synthetic_accessibility = true_synthetic_accessibility_scores(sani_mols).reshape(-1,1)  # vec - [-5, 5]
    drug_candidacy = drug_candidate_scores(water_octanol_partition, 
        synthetic_accessibility, novelty).reshape(-1,1)  # vec - [0,1]

    scores = np.hstack((np_score, water_octanol_partition, synthetic_accessibility, qed, drug_candidacy))
    nuvd = np.array([novelty, uniqueness, validity, diversity])

    return nuvd, scores


def score_model(path, num_expt=1, return_properties=False):
    '''
    Takes the path to a pre-trained model checkpoint, generates 1000 molecules, and scores them

    #FIXME!!! For some reason this function returns (nuvd, avg_scores, nuvd, avg_scores)
    Not entirely sure why, but temporary usage: nuvd, avg_scores,_,_ = score_model(path)

    Returns:
    nuvd: [novelty, uniqueness, validity, diversity]
    avg_scores: avg[np likeness, logP, SA, QED, drug candidacy]
    properties: per molecule properties (if return_properties=True)
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
                               noise_scale=0.6
                               )
    gen = MoFlow(model_params_gflow)
    chk = torch.load(path)
    gen.load_state_dict(chk['GStateDict'])

    nuvds = np.zeros((1,4))
    avgs = np.zeros((1,5))

    for i in range(num_expt):
        adj, x = generate_mols(gen, batch_size=1000)
        nuvd, properties = evaluate_scores(adj, x)
        avg_scores = np.mean(properties, axis=0).flatten()
        nuvds = np.vstack((nuvds, nuvd))
        avgs = np.vstack((avgs, avg_scores))
    
    nuvds = nuvds[1:, :].mean(axis=0).tolist()
    avgs = avgs[1:, :].mean(axis=0).tolist()

    if return_properties: return nuvds, avgs, properties
    else: return nuvds, avgs
