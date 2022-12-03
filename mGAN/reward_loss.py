import numpy as np
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer as sa # SA Scoring from Ertl
from rdkit.Contrib.NP_Score import npscorer as natp # natural product likeliness
from rdkit.Chem import QED
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from mflow.utils.molecular_metrics import *
from mflow.models.utils import construct_mol, construct_mol_with_validation
from data.sparse_molecular_dataset import SparseMolecularDataset

train_sparse= SparseMolecularDataset()
train_sparse.load('./data/qm9_5k.sparsedataset')
fscore = natp.readNPModel()

def normalize(data):
    '''
    data is a numpy array that gets normalized to [0, 1]
    '''
    if np.max(data) == np.min(data): return data / np.max(data)
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def synthetic_accessibility_scores(sanitized_mols):
    '''
    Synthetic acessability score
    Originally: [1, 10] - lower is better
    Modified: [0.1, 1] - higher is better
    
    http://www.jcheminf.com/content/1/1/8
    '''
    scores = np.array([-sa.calculateScore(mol) if mol is not None 
                       else -10 for mol in sanitized_mols])  # [-10, -1]
    return (scores + 11) / 10  # [0, 1]


def natural_product_scores(sanitized_mols):
    '''
    Ertl's natural product likness scores
    Originally: [-5, 5] - higher is better
    Modified: [0, 1] - higher is better
    
    http://pubs.acs.org/doi/abs/10.1021/ci700286x
    '''
    scores = np.array([natp.scoreMol(mol, fscore) if mol is not None 
                       else -5 for mol in sanitized_mols])  # [-5, 5]
    return (scores + 5) / 10  # [0, 1]


def drug_candidate_scores(logP, syn_acc, nov):
    scores = (constant_bump(logP, 0.210, 0.945) + syn_acc + nov + (1 - nov) * 0.3) / 4
    return scores # open ended


def constant_bump(x, x_low, x_high, decay=0.025):
    return np.select(condlist=[x <= x_low, x >= x_high],
                        choicelist=[np.exp(- (x - x_low) ** 2 / decay),
                                    np.exp(- (x - x_high) ** 2 / decay)],
                        default=np.ones_like(x))


def calculate_rewards(edges, nodes, atomic_num_list, training_data=train_sparse, weights=None):
    """
    
    In:
    edges, nodes: adjacency and label tensors/matrices corresponding to molecules
    atomic_number_list: decoding key for elements of atoms
    training_data: dataset used to evaluate novelty and diversity (right now just a sparse subset)
    weights: vector of length 7 used to weigh molecule scores. Orderd:
    [np_score, water_octanol_partition, synthetic_accessibility, novelty, uniqueness, diversity, validity]

    Out:
    batch-sized vector in which each entry is the product of the compounds individual scores
    """    
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

    adj = edges.cpu().__array__()  # (bs,4,9,9)
    x = nodes.cpu().__array__()  # (bs,9,5)
            
    mols = list(map(to_mol, zip(adj, x)))
    valid_mols = list(map(to_validated_mol, zip(adj, x)))
    flags = list(map(clean_sanitize, valid_mols))  # see clean_sanitize spec
    sani_mols = [mol if validity == 1 else None for mol, validity in zip(valid_mols, flags)]

    mm = MolecularMetrics()
    water_octanol_partition = mm.water_octanol_partition_coefficient_scores(mols, norm=True).flatten().reshape(-1,1) # vec - [0, 1]
    novelty = mm.novel_scores(mols, training_data).flatten().reshape(-1,1)  # vec - [0, 1]
    uniqueness = mm.unique_scores(mols).flatten().reshape(-1,1)  # vec - [0, 1]
    validity = mm.valid_scores(mols).flatten().reshape(-1,1)  # vec - [0, 1]
    qed = mm.quantitative_estimation_druglikeness_scores(mols, norm=True).flatten().reshape(-1,1)  # vec - [0, 1]

    np_score = natural_product_scores(sani_mols).reshape(-1,1)  # vec - [0, 1]
    synthetic_accessibility = synthetic_accessibility_scores(sani_mols).reshape(-1,1)  # vec - [0, 1]
    diversity = mm.diversity_scores(sani_mols, training_data).reshape(-1,1) # vec - [0, 1]
    # drug_candidacy = drug_candidate_scores(water_octanol_partition, 
        # synthetic_accessibility, novelty).reshape(-1,1)  # vec - [0,1]

    scores = np.hstack((np_score, water_octanol_partition, synthetic_accessibility, 
                        novelty, uniqueness, diversity, validity, qed))


    if weights != None:
        weights= np.array(weights).flatten()
        weights= np.broadcast_to(weights, scores.shape)
        scores = np.multiply(scores, weights)
    
    return scores.prod(axis=1).flatten()


def reward_from_smiles(smiles):
    '''currently meant to handle one smiles at a time'''
    nodes, edges = smiles_to_nodes_edges(smile)
    return calculate_rewards(torch.unsqueeze(edges,dim=0), torch.unsqueeze(nodes,dim=0), atomic_num_list) # what is atomic_num_list
    
