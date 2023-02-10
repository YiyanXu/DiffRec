import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
from torch.nn.init import xavier_normal_, constant_, xavier_uniform_
from kmeans_pytorch import kmeans


class AutoEncoder(nn.Module):
    """
    Guassian Diffusion for large-scale recommendation.
    """
    def __init__(self, item_emb, n_cate, in_dims, out_dims, device, act_func, reparam=True, dropout=0.1):
        super(AutoEncoder, self).__init__()

        self.item_emb = item_emb
        self.n_cate = n_cate
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.act_func = act_func
        self.n_item = len(item_emb)
        self.reparam = reparam
        self.dropout = nn.Dropout(dropout)

        if n_cate == 1:  # no clustering
            in_dims_temp = [self.n_item] + self.in_dims[:-1] + [self.in_dims[-1] * 2]
            out_dims_temp = [self.in_dims[-1]] + self.out_dims + [self.n_item]

            encoder_modules = []
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:]):
                encoder_modules.append(nn.Linear(d_in, d_out))
                if self.act_func == 'relu':
                    encoder_modules.append(nn.ReLU())
                elif self.act_func == 'sigmoid':
                    encoder_modules.append(nn.Sigmoid())
                elif self.act_func == 'tanh':
                    encoder_modules.append(nn.Tanh())
                else:
                    raise ValueError
            self.encoder = nn.Sequential(*encoder_modules)

            decoder_modules = []
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:]):
                decoder_modules.append(nn.Linear(d_in, d_out))
                if self.act_func == 'relu':
                    decoder_modules.append(nn.ReLU())
                elif self.act_func == 'sigmoid':
                    decoder_modules.append(nn.Sigmoid())
                elif self.act_func == 'tanh':
                    decoder_modules.append(nn.Tanh())
                elif self.act_func == 'leaky_relu':
                    encoder_modules.append(nn.LeakyReLU())
                else:
                    raise ValueError
            decoder_modules.pop()
            self.decoder = nn.Sequential(*decoder_modules)
        
        else:
            self.cluster_ids, _ = kmeans(X=item_emb, num_clusters=n_cate, distance='euclidean', device=device)
            # cluster_ids(labels): [0, 1, 2, 2, 1, 0, 0, ...]
            category_idx = []
            for i in range(n_cate):
                idx = np.argwhere(self.cluster_ids.numpy() == i).squeeze().tolist()
                category_idx.append(torch.tensor(idx, dtype=int))
            self.category_idx = category_idx  # [cate1: [iid1, iid2, ...], cate2: [iid3, iid4, ...], cate3: [iid5, iid6, ...]]
            self.category_map = torch.cat(tuple(category_idx), dim=-1)  # map
            self.category_len = [len(self.category_idx[i]) for i in range(n_cate)]  # item num in each category
            print("category length: ", self.category_len)
            assert sum(self.category_len) == self.n_item

            ##### Build the Encoder and Decoder #####
            encoder_modules = [[] for _ in range(n_cate)]
            decode_dim = []
            for i in range(n_cate):
                if i == n_cate - 1:
                    latent_dims = list(self.in_dims - np.array(decode_dim).sum(axis=0))
                else:
                    latent_dims = [int(self.category_len[i] / self.n_item * self.in_dims[j]) for j in range(len(self.in_dims))]
                    latent_dims = [latent_dims[j] if latent_dims[j] != 0 else 1 for j in range(len(self.in_dims))]
                in_dims_temp = [self.category_len[i]] + latent_dims[:-1] + [latent_dims[-1] * 2]
                decode_dim.append(latent_dims)
                for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:]):
                    encoder_modules[i].append(nn.Linear(d_in, d_out))
                    if self.act_func == 'relu':
                        encoder_modules[i].append(nn.ReLU())
                    elif self.act_func == 'sigmoid':
                        encoder_modules[i].append(nn.Sigmoid())
                    elif self.act_func == 'tanh':
                        encoder_modules[i].append(nn.Tanh())
                    elif self.act_func == 'leaky_relu':
                        encoder_modules[i].append(nn.LeakyReLU())
                    else:
                        raise ValueError

            self.encoder = nn.ModuleList([nn.Sequential(*encoder_modules[i]) for i in range(n_cate)])
            print("Latent dims of each category: ", decode_dim)

            self.decode_dim = [decode_dim[i][::-1] for i in range(len(decode_dim))]

            if len(out_dims) == 0:  # one-layer decoder: [encoder_dim_sum, n_item]
                out_dim = self.in_dims[-1]
                decoder_modules = []
                decoder_modules.append(nn.Linear(out_dim, self.n_item))
                self.decoder = nn.Sequential(*decoder_modules)
            else:  # multi-layer decoder: [encoder_dim, hidden_size, cate_num]
                decoder_modules = [[] for _ in range(n_cate)]
                for i in range(n_cate):
                    out_dims_temp = self.decode_dim[i] + [self.category_len[i]]
                    for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:]):
                        decoder_modules[i].append(nn.Linear(d_in, d_out))
                        if self.act_func == 'relu':
                            decoder_modules[i].append(nn.ReLU())
                        elif self.act_func == 'sigmoid':
                            decoder_modules[i].append(nn.Sigmoid())
                        elif self.act_func == 'tanh':
                            decoder_modules[i].append(nn.Tanh())
                        elif self.act_func == 'leaky_relu':
                            encoder_modules[i].append(nn.LeakyReLU())
                        else:
                            raise ValueError
                    decoder_modules[i].pop()
                self.decoder = nn.ModuleList([nn.Sequential(*decoder_modules[i]) for i in range(n_cate)])
            
        self.apply(xavier_normal_initialization)
        
    def Encode(self, batch):
        batch = self.dropout(batch)
        if self.n_cate == 1:
            hidden = self.encoder(batch)
            mu = hidden[:, :self.in_dims[-1]]
            logvar = hidden[:, self.in_dims[-1]:]

            if self.training and self.reparam:
                latent = self.reparamterization(mu, logvar)
            else:
                latent = mu
            
            kl_divergence = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

            return batch, latent, kl_divergence

        else: 
            batch_cate = []
            for i in range(self.n_cate):
                batch_cate.append(batch[:, self.category_idx[i]])
            # [batch_size, n_items] -> [[batch_size, n1_items], [batch_size, n2_items], [batch_size, n3_items]]
            latent_mu = []
            latent_logvar = []
            for i in range(self.n_cate):
                hidden = self.encoder[i](batch_cate[i])
                latent_mu.append(hidden[:, :self.decode_dim[i][0]])
                latent_logvar.append(hidden[:, self.decode_dim[i][0]:])
            # latent: [[batch_size, latent_size1], [batch_size, latent_size2], [batch_size, latent_size3]]

            mu = torch.cat(tuple(latent_mu), dim=-1)
            logvar = torch.cat(tuple(latent_logvar), dim=-1)
            if self.training and self.reparam:
                latent = self.reparamterization(mu, logvar)
            else:
                latent = mu

            kl_divergence = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

            return torch.cat(tuple(batch_cate), dim=-1), latent, kl_divergence
    
    def reparamterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def Decode(self, batch):
        if len(self.out_dims) == 0 or self.n_cate == 1:  # one-layer decoder
            return self.decoder(batch)
        else:
            batch_cate = []
            start=0
            for i in range(self.n_cate):
                end = start + self.decode_dim[i][0]
                batch_cate.append(batch[:, start:end])
                start = end
            pred_cate = []
            for i in range(self.n_cate):
                pred_cate.append(self.decoder[i](batch_cate[i]))
            pred = torch.cat(tuple(pred_cate), dim=-1)

            return pred
    
def compute_loss(recon_x, x):
    return -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))  # multinomial log likelihood in MultVAE


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
                