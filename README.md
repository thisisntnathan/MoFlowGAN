# MoFlowGAN

You've found the working repo for MoFlowGAN, a normalizing flow that can be trained like a GAN to generate high quality molecular graphs. The code here works, so feel free to train your own model while we put a last minute shine on our preprint!  

## Training your own model

### Installing dependencies

It's best to run MoFlowGAN in its own conda environment. Learn more about conda and virtual enviornments [here](https://conda.io/projects/conda/en/latest/index.html)!

```
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

Alternatively, you can just clone this repository using:

```
git clone https://github.com/thisisntnathan/MoFlowGAN.git MoFlowGAN
```

<!-- We should probably host a copy of the kekulized datasets people can just wget? -->

### Training

To train MoFlowGAN call `reward_trainer.py` using the command below making sure to set your desired objective coefficients (`adv_reg` and `rl_reg`). There are many more hyperparameters you can play with, a full list of arguments see `reward_trainer.py`

```
python reward_trainer.py --data_name qm9 -t 237 --max_epochs 50 --gpu 0 --adv_reg 0.27 --rl_reg 0.42 --debug True --save_epochs 5 --save_dir=./results/ 2>&1 | tee ./results/MoFlowGAN.log
```

### Evaluation

We provide a simple jupyter notebook `evaluate.ipynb` for evaluating your model's performance!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thisisntnathan/MoFlowGAN/blob/main/evaluate.ipynb)

## Contribution

If you have any questions, comments, or suggestions feel free to [reach out](mailto:nml64@cornell.edu) (or submit a PR)!

## Acknowledgements

MoFlowGAN started off as final project for Cornell's CS 6784: Advanced Topics in Machine Learning - Deep Learning. We thank [Prof. Killian Weinberger](https://www.cs.cornell.edu/~kilian/) for insights, input, and fruitful discussion.  

MoFlowGAN's base layers are structured off those of [MoFlow](https://arxiv.org/abs/2006.10137) ([code](https://github.com/calvin-zcx/moflow)) and [MolGAN](https://arxiv.org/abs/1805.11973) ([code](https://github.com/nicola-decao/MolGAN))
