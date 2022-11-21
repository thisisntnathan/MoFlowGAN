import torch
from mGAN.models import Discriminator 

 
# generate test adjacency and node feature tensors in format from Nathan's 
# generator
batch_size = 100
adj = torch.rand(batch_size,4,9,9)
node =  torch.rand(batch_size,9,5)
 
 
# define parameters for discriminator and reward networks
conv_dim  = [[128, 64], 128, [128, 64]] # default values from molgan
m_dim = 5 # num of atom types
b_dim = 5 #  supposed to be num of bond types but .... it isnt...
dropout  = 0 # dropout probability
 
# construct discriminator
disc = Discriminator(conv_dim, m_dim, b_dim, dropout)
 
# construct reward network that predicts molecule performance
reward = Discriminator(conv_dim, m_dim, b_dim, dropout)
 
 
# make prediction for discriminator
 
# note this will output a number between (-infty,infty)
# this will output scores/logits for discriminator
# to get probabilities between (0,1) pass argument activation = torch.sigmoid
# as seen in prediction made by reward network below
# probably do not want to do this in training discriminator as it will be 
# numerically unstable versus working with loss function that takes in 
# scores/logits directly
# can pass argument activation like
 
discriminator_logits = disc(adj.permute(0,2,3,1), None, node)
 
 
# make prediction for reward network
 
# note I use sigmoid activation here. MolGAN says they use sigmoid activation
# for reward network, this would be appropriate if the performance metrics we
# predict fall in some bounded range
 
reward_predictions = reward(adj.permute(0,2,3,1), None, node, 
                            activation = torch.sigmoid)
