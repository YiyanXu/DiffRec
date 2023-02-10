import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
from torch.nn.init import xavier_normal_, constant_, xavier_uniform_

class DNN(nn.Module):
    """
    A deep neural network for the reverse process of latent diffusion.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, act_func='tanh', dropout=0.5):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_emb_dim = emb_size
        self.time_type = time_type
        self.norm = norm
        
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims

        self.in_modules = []
        for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:]):
            self.in_modules.append(nn.Linear(d_in, d_out))
            if act_func == 'tanh':
                self.in_modules.append(nn.Tanh())
            elif act_func == 'relu':
                self.in_modules.append(nn.ReLU())
            elif act_func == 'sigmoid':
                self.in_modules.append(nn.Sigmoid())
            elif act_func == 'leaky_relu':
                self.in_modules.append(nn.LeakyReLU())
            else:
                raise ValueError
        self.in_layers = nn.Sequential(*self.in_modules)

        self.out_modules = []
        for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:]):
            self.out_modules.append(nn.Linear(d_in, d_out))
            if act_func == 'tanh':
                self.out_modules.append(nn.Tanh())
            elif act_func == 'relu':
                self.out_modules.append(nn.ReLU())
            elif act_func == 'sigmoid':
                self.out_modules.append(nn.Sigmoid())
            elif act_func == 'leaky_relu':
                self.out_modules.append(nn.LeakyReLU())
            else:
                raise ValueError
        self.out_modules.pop()
        self.out_layers = nn.Sequential(*self.out_modules)

        self.dropout = nn.Dropout(dropout)

        self.apply(xavier_normal_initialization)
    
    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x, dim=-1)
        x = self.dropout(x)
        h = torch.cat([x, emb], dim=-1)
        h = self.in_layers(h)
        h = self.out_layers(h)

        return h

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def xavier_normal_initialization(module):
    r""" using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.
    .. _`xavier_normal_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_
    Examples:
        >>> self.apply(xavier_normal_initialization)
    """
    if isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)         



