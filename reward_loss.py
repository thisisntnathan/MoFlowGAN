from mflow.utils.molecular_metrics import *
from mflow.models.utils import construct_mol, construct_mol_with_validation


from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer as sa # SA Scoring from Ertl
from rdkit.Contrib.NP_Score import npscorer as natp # natural product likeliness
from rdkit.Chem import QED

fscore = natp.readNPModel()

def synthetic_accessibility_total_score(sanitized_mols):
    scores = list(map(sa.calculateScore,sanitized_mols))
    return sum(scores)/len(scores)

def natural_product_total_score(sanitized_mols):
    scores = [natp.scoreMol(e, fscore) for e in sanitized_mols]
    return sum(scores)/len(scores)

def qed_total_score(sanitized_mols):
    scores = list(map(QED.default,sanitized_mols))
    return sum(scores)/len(scores)

def reward_loss(edges,nodes, atomic_num_list, training_data=None, weights = None):
    """
    weights is an iterable of numbers of length 7 corresponding to
    [validity,uniqueness,novelty,drug_canditate_scores,synthetic_accessibility,diversity]

    Returns sum of scores

    """
    adj = edges.__array__()  # , gpu)  (1000,4,9,9)
    x = nodes.__array__()  # , gpu)  (1000,9,5)
    def to_mol(adj_x):
        adj_elem, x_elem = adj_x
        return construct_mol(x_elem, adj_elem, atomic_num_list)
    
    def to_validated_mol(adj_x):
        adj_elem, x_elem = adj_x
        return construct_mol_with_validation(x_elem, adj_elem, atomic_num_list)

    mols = list(map(to_mol,zip(adj,x)))
    valid_mols = list(map(to_validated_mol,zip(adj,x)))
    
    sanitized_flags = list(map(Chem.SanitizeMol,valid_mols))

    m = MolecularMetrics()
    validity = m.valid_total_score(mols)
    uniqueness = m.unique_total_score(mols)
    #water_octanol_partition_coef = m.water_octanol_partition_coefficient_scores(mols).mean() # range -2.12178879609, 6.0429063424
    synthetic_accessibility = synthetic_accessibility_total_score(valid_mols)
    if training_data != None:
        novelty = m.novel_total_score(mols, training_data)
        drug_canditate_scores = m.drugcandidate_scores(valid_mols,training_data)
    else:
        novelty = 0
        drug_canditate_scores = 0
    diversity = m.diversity_scores(mols,training_data).mean()

    scores = [validity,uniqueness,novelty,drug_canditate_scores,synthetic_accessibility,diversity]

    if weights == None:
        return sum(scores)
    else:
        return sum([score*weight for score, weight in zip(scores, weights)])
