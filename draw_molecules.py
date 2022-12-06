import os
from rdkit.Chem import Draw

from mflow.models.utils import check_validity
from mflow.generate import generate_mols
from model_eval import load_model

def show_and_tell(path):
    '''
    Generate 100 molecules using the model and draw them
    '''
    gen = load_model(path)
    
    atomic_num_list=[6, 7, 8, 9, 0]

    adj, x = generate_mols(gen, batch_size=100)
    val_res = check_validity(adj, x, atomic_num_list)
    
    pth = os.path.normpath(os.path.dirname(path)).split(os.sep)
    gen_dir = '/notebooks/results/validation/generated'
    os.makedirs(gen_dir, exist_ok=True)
    filepath = os.path.join(gen_dir, 'generated_mols_{}.png'.format(pth[-1]))
    img = Draw.MolsToGridImage(val_res['valid_mols'], legends=val_res['valid_smiles'], 
                               molsPerRow=10, subImgSize=(300, 300), returnPNG=False)
    img.save(filepath)
