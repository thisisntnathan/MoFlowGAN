# MoFlowGAN

You've found the working repo for MoFlowGAN, a normalizing flow that can be trained like a GAN to generate high quality molecular graphs.  
The code here works, so feel free to train your own model while we put a last minute shine on things before we release our preprint!  

## Training your own model

### Installing dependencies

It's best to run MoFlowGAN in its own conda environment. Learn more about conda and virtual enviornments [here](https://conda.io/projects/conda/en/latest/index.html)!

```bash
conda create --name moflow python pandas matplotlib 
conda activate moflow
conda install pytorch=1.12 torchvision cudatoolkit -c pytorch
conda install rdkit
conda install orderedset
conda install tabulate
conda install networkx
conda install scipy
conda install seaborn
pip install cairosvg
pip install tqdm
```

### Preprocess the data

Right now we've only trained MoFlowGAN on [QM9](http://quantum-machine.org/datasets/); our plans are to extend to ZINC250.  

```
cd data
python data_preprocess.py --data_name qm9
```

### Training

To train MoFlowGAN call `reward_trainer.py` using the command below making sure to set your desired objective coefficients (`adv_reg` and `rl_reg`). There are many more hyperparameters you can play with, a full list of arguments see `reward_trainer.py`

```
python reward_trainer.py --data_name qm9 -t 237 --max_epochs 50 --gpu 0 --adv_reg 0.27 --rl_reg 0.42 --debug True --save_epochs 5 --save_dir=./results/ 2>&1 | tee ./results/MoFlowGAN.log
```

### Evaluation

We provide a simple jupyter notebook `eval.ipynb` for evaluating your models!

<!-- TODO: Clean up a notebook for production -->

## Acknowledgements

MoFlowGANs base layers are structured off those of [MoFlow](https://arxiv.org/abs/2006.10137) ([code](https://github.com/calvin-zcx/moflow)) and [MolGAN](https://arxiv.org/abs/1805.11973) ([code](https://github.com/nicola-decao/MolGAN))
