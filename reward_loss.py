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
    scores = [sa.calculateScore(mol) if mol is not None else None for mol in sanitized_mols]  # [None | 10, 1]
    scores = np.array(list(map(lambda x: -10 if x is None else -x, scores)))  # [-10, -1]
    return normalize(scores)  # [0, 1]


def natural_product_scores(sanitized_mols):
    scores = np.array([natp.scoreMol(mol, fscore) if mol is not None 
                       else None for mol in sanitized_mols])
    return normalize(scores)  # [0, 1]


# def qed_scores(sanitized_mols):
#     scores = np.array(list(map(QED.default, sanitized_mols)))
#     return normalize(scores)  # [0, 1]


def drug_candidate_scores(logP, syn_acc, nov):
    scores = (constant_bump(logP, 0.210, 0.945) + syn_acc + nov + (1 - nov) * 0.3) / 4
    return normalize(scores)


def constant_bump(x, x_low, x_high, decay=0.025):
    return np.select(condlist=[x <= x_low, x >= x_high],
                        choicelist=[np.exp(- (x - x_low) ** 2 / decay),
                                    np.exp(- (x - x_high) ** 2 / decay)],
                        default=np.ones_like(x))


def calculate_rewards(edges, nodes, atomic_num_list, training_data=train_sparse, weights=None):
    """
    weights is an iterable of numbers of length 8 corresponding to
    [np_score, water_octanol_partition, synthetic_accessibility, novelty, drug_candidacy, uniqueness, diversity, validity]

    Returns a batch-sized vector in which each entry is the product of the compounds individual scores
    """
    adj = edges.cpu().__array__()  # (bs,4,9,9)
    x = nodes.cpu().__array__()  # (bs,9,5)

    def to_mol(adj_x):
        adj_elem, x_elem = adj_x
        return construct_mol(x_elem, adj_elem, atomic_num_list)
    
    def to_validated_mol(adj_x):
        adj_elem, x_elem = adj_x
        return construct_mol_with_validation(x_elem, adj_elem, atomic_num_list)
    
    # def clean_sanitize(mol):
    #     '''
    #     There's an issue with trying to sanitize fake molecules
    #     Solution adapted from: https://github.com/rdkit/rdkit/issues/2216
    #     '''
    #     try:
    #         Chem.SanitizeMol(mol)
    #     except:
    #         try:
    #             Chem.FastFindRings(mol)  # this is not Chem.FindFastRings()
    #             mol.UpdatePropertyCache()
    #             mol.calcImplicitValence()
    #         except:
    #             return None
            

    mols = list(map(to_mol, zip(adj, x)))
    valid_mols = list(map(to_validated_mol, zip(adj, x)))

    # valid_mols = mols
    # for mol in valid_mols:
    #     try: Chem.SanitizeMol(mol)
    #     except: 
    #         try: mol.calcImplicitValence()
    #         except: 
    #             try: Chem.FastFindRings(mol)  # this is not Chem.FindFastRings()
    #             except: 
    #                 try: mol.UpdatePropertyCache()
    #                 except: mol = None
    # sanitized_flags = list(map(clean_sanitize, valid_mols))

    mm = MolecularMetrics()
    # np_score = natural_product_scores(valid_mols).reshape(-1,1)  # vec - [0, 1]
    water_octanol_partition = mm.water_octanol_partition_coefficient_scores(mols, norm=True).reshape(-1,1) # vec - [0, 1]
    # synthetic_accessibility = synthetic_accessibility_scores(valid_mols).reshape(-1,1)  # vec - [0, 1]
    novelty = mm.novel_scores(mols, training_data).reshape(-1,1)  # vec - [0, 1]
    uniqueness = mm.unique_scores(mols).reshape(-1,1)  # vec - [0, 1]
    # diversity = mm.diversity_scores(valid_mols, training_data).reshape(-1,1) # vec - [0, 1]
    validity = mm.valid_scores(mols).reshape(-1,1)  # vec - [0, 1]
    qed = mm.quantitative_estimation_druglikeness_scores(mols, norm=True).reshape(-1,1)  # vec - [0, 1]
    # drug_candidacy = drug_candidate_scores(water_octanol_partition, 
        # synthetic_accessibility, novelty).reshape(-1,1)  # vec - [0,1]

    # scores = np.hstack((np_score, water_octanol_partition, synthetic_accessibility, 
    #                     novelty, drug_candidacy, uniqueness, diversity, validity, qed))
    scores = np.hstack((water_octanol_partition, novelty, uniqueness, validity, qed))

    if weights != None:
        weights= np.array(weights).flatten()
        weights= np.broadcast_to(weights, scores.shape)
        scores = np.multiply(scores, weights)
    
    return scores.prod(axis=1).flatten()
