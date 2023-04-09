"""
Train a diffusion model for recommendation
"""

import argparse
from ast import parse
import os
import time
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import scipy.sparse as sp

import models.gaussian_diffusion as gd
from models.Autoencoder import AutoEncoder as AE
from models.Autoencoder import compute_loss
from models.DNN import DNN
import evaluate_utils
import data_utils
from copy import deepcopy

import random
random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) #gpu
np.random.seed(random_seed) #numpy
random.seed(random_seed) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='yelp_clean', help='choose the dataset')
parser.add_argument('--data_path', type=str, default='../datasets/', help='load data path')
parser.add_argument('--emb_path', type=str, default='../datasets/')
parser.add_argument('--lr1', type=float, default=0.0001, help='learning rate for Autoencoder')
parser.add_argument('--lr2', type=float, default=0.0001, help='learning rate for MLP')
parser.add_argument('--wd1', type=float, default=0.0, help='weight decay for Autoencoder')
parser.add_argument('--wd2', type=float, default=0.0, help='weight decay for MLP')
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
parser.add_argument('--log_name', type=str, default='log', help='the log name')
parser.add_argument('--round', type=int, default=1, help='record the experiment')

# params for the Autoencoder
parser.add_argument('--n_cate', type=int, default=3, help='category num of items')
parser.add_argument('--in_dims', type=str, default='[300]', help='the dims for the encoder')
parser.add_argument('--out_dims', type=str, default='[]', help='the hidden dims for the decoder')
parser.add_argument('--act_func', type=str, default='tanh', help='activation function for autoencoder')
parser.add_argument('--lamda', type=float, default=0.03, help='hyper-parameter of multinomial log-likelihood for AE: 0.01, 0.02, 0.03, 0.05')
parser.add_argument('--optimizer1', type=str, default='AdamW', help='optimizer for AE: Adam, AdamW, SGD, Adagrad, Momentum')
parser.add_argument('--anneal_cap', type=float, default=0.005)
parser.add_argument('--anneal_steps', type=int, default=500)
parser.add_argument('--vae_anneal_cap', type=float, default=0.3)
parser.add_argument('--vae_anneal_steps', type=int, default=200)
parser.add_argument('--reparam', type=bool, default=True, help="Autoencoder with variational inference or not")

parser.add_argument('--w_min', type=float, default=0.1, help='the minimum weight for interactions')
parser.add_argument('--w_max', type=float, default=1., help='the maximum weight for interactions')

# params for the MLP
parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('--mlp_dims', type=str, default='[300]', help='the dims for the DNN')
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')
parser.add_argument('--mlp_act_func', type=str, default='tanh', help='the activation function for MLP')
parser.add_argument('--optimizer2', type=str, default='AdamW', help='optimizer for MLP: Adam, AdamW, SGD, Adagrad, Momentum')

# params for diffusion
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=5, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.0001)
parser.add_argument('--noise_max', type=float, default=0.02)
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=10, help='steps for sampling/denoising')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')

args = parser.parse_args()
print("args:", args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if args.cuda else "cpu")

print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

### DATA LOAD ###
train_path = args.data_path + 'train_list.npy'
valid_path = args.data_path + 'valid_list.npy'
test_path = args.data_path + 'test_list.npy'

train_data, train_data_ori, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(train_path, valid_path, test_path, args.w_min, args.w_max)
train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

if args.tst_w_val:
    tv_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A) + torch.FloatTensor(valid_y_data.A))
    test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)
mask_tv = train_data_ori + valid_y_data

print('data ready.')

### Build Autoencoder ###
emb_path = args.emb_path + args.dataset + '/item_emb.npy'
item_emb = torch.from_numpy(np.load(emb_path, allow_pickle=True))
assert len(item_emb) == n_item
out_dims = eval(args.out_dims)
in_dims = eval(args.in_dims)[::-1]
Autoencoder = AE(item_emb, args.n_cate, in_dims, out_dims, device, args.act_func, args.reparam).to(device)

### Build Gaussian Diffusion ###
if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError("Unimplemented mean type %s" % args.mean_type)

diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, \
        args.noise_scale, args.noise_min, args.noise_max, args.steps, device).to(device)

### Build MLP ###
latent_size = in_dims[-1]
mlp_out_dims = eval(args.mlp_dims) + [latent_size]
mlp_in_dims = mlp_out_dims[::-1]
model = DNN(mlp_in_dims, mlp_out_dims, args.emb_size, time_type="cat", norm=args.norm, act_func=args.mlp_act_func).to(device)

param_num = 0
AE_num = sum([param.nelement() for param in Autoencoder.parameters()])
mlp_num = sum([param.nelement() for param in model.parameters()])
diff_num = sum([param.nelement() for param in diffusion.parameters()])  # 0
param_num = AE_num + mlp_num + diff_num
print("Number of parameters:", param_num)

if args.optimizer1 == 'Adagrad':
    optimizer1 = optim.Adagrad(
        model.parameters(), lr=args.lr1, initial_accumulator_value=1e-8, weight_decay=args.wd1)
elif args.optimizer1 == 'Adam':
    optimizer1 = optim.Adam(Autoencoder.parameters(), lr=args.lr1, weight_decay=args.wd1)
elif args.optimizer1 == 'AdamW':
    optimizer1 = optim.AdamW(Autoencoder.parameters(), lr=args.lr1, weight_decay=args.wd1)
elif args.optimizer1 == 'SGD':
    optimizer1 = optim.SGD(Autoencoder.parameters(), lr=args.lr1, weight_decay=args.wd1)
elif args.optimizer1 == 'Momentum':
    optimizer1 = optim.SGD(Autoencoder.parameters(), lr=args.lr1, momentum=0.95, weight_decay=args.wd1)

if args.optimizer2 == 'Adagrad':
    optimizer2 = optim.Adagrad(
        model.parameters(), lr=args.lr2, initial_accumulator_value=1e-8, weight_decay=args.wd2)
elif args.optimizer2 == 'Adam':
    optimizer2 = optim.Adam(model.parameters(), lr=args.lr2, weight_decay=args.wd2)
elif args.optimizer2 == 'AdamW':
    optimizer2 = optim.AdamW(model.parameters(), lr=args.lr2, weight_decay=args.wd2)
elif args.optimizer2 == 'SGD':
    optimizer2 = optim.SGD(model.parameters(), lr=args.lr2, weight_decay=args.wd2)
elif args.optimizer2 == 'Momentum':
    optimizer2 = optim.SGD(model.parameters(), lr=args.lr2, momentum=0.95, weight_decay=args.wd2)
print("models ready.")


def evaluate(data_loader, data_te, mask_his, topN):
    model.eval()
    Autoencoder.eval()
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]

    predict_items = []
    target_items = []
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())
    
    if args.n_cate > 1:
        category_map = Autoencoder.category_map.to(device)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device)

            # mask map
            his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]]

            _, batch_latent, _ = Autoencoder.Encode(batch)
            batch_latent_recon = diffusion.p_sample(model, batch_latent, args.sampling_steps, args.sampling_noise)
            prediction = Autoencoder.Decode(batch_latent_recon)  # [batch_size, n1_items + n2_items + n3_items]

            prediction[his_data.nonzero()] = -np.inf  # mask ui pairs in train & validation set

            _, mapped_indices = torch.topk(prediction, topN[-1])  # topk category idx

            if args.n_cate > 1:
                indices = category_map[mapped_indices]
            else:
                indices = mapped_indices

            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)

    test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)

    return test_results

best_recall, best_epoch = -100, 0
best_test_result = None
update_count = 0
update_count_vae = 0
save_path = args.save_path + args.dataset + '/'

if args.n_cate > 1:
    start_time = time.time()
    category_map = Autoencoder.category_map.cpu().numpy()
    reverse_map = {category_map[i]:i for i in range(len(category_map))}
    # mask for validation: train_dataset
    mask_idx_train = list(train_data.nonzero())
    mapped_mask_iid_train = np.array([reverse_map[mask_idx_train[1][i]] for i in range(len(mask_idx_train[0]))])
    mask_train = sp.csr_matrix((np.ones_like(mask_idx_train[0]), (mask_idx_train[0], mapped_mask_iid_train)), \
        dtype='float64', shape=(n_user, n_item))

    # mask for test: train_dataset + valid_dataset
    mask_idx_val = list(valid_y_data.nonzero())  # valid dataset
    mapped_mask_iid_val = np.array([reverse_map[mask_idx_val[1][i]] for i in range(len(mask_idx_val[0]))])
    mask_val = sp.csr_matrix((np.ones_like(mask_idx_val[0]), (mask_idx_val[0], mapped_mask_iid_val)), \
        dtype='float64', shape=(n_user, n_item))

    mask_tv = mask_train + mask_val

    print("Preparing mask for validation & test costs " + time.strftime(
                            "%H: %M: %S", time.gmtime(time.time()-start_time)))
else:
    mask_train = train_data

print("Start training...")

for epoch in range(1, args.epochs + 1):
    if epoch - best_epoch >= 20:
        print('-'*18)
        print('Exiting from training early')
        break

    Autoencoder.train()
    model.train()

    start_time = time.time()

    batch_count = 0
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        batch_count += 1
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        batch_cate, batch_latent, vae_kl = Autoencoder.Encode(batch)

        terms = diffusion.training_losses(model, batch_latent, args.reweight)
        elbo = terms["loss"].mean()  # loss from diffusion
        batch_latent_recon = terms["pred_xstart"]

        batch_recon = Autoencoder.Decode(batch_latent_recon)

        if args.anneal_steps > 0:
            lamda = max((1. - update_count / args.anneal_steps) * args.lamda, args.anneal_cap)
        else:
            lamda = max(args.lamda, args.anneal_cap)
        
        if args.vae_anneal_steps > 0:
            anneal = min(args.vae_anneal_cap, 1. * update_count_vae / args.vae_anneal_steps)
        else:
            anneal = args.vae_anneal_cap

        vae_loss = compute_loss(batch_recon, batch_cate) + anneal * vae_kl  # loss from autoencoder
        
        if args.reweight:
            loss = lamda * elbo + vae_loss
        else:
            loss = elbo + lamda * vae_loss
        
        update_count_vae += 1

        total_loss += loss
        loss.backward()
        optimizer1.step()
        optimizer2.step()

    update_count += 1
    
    if epoch % 5 == 0:
        valid_results = evaluate(test_loader, valid_y_data, mask_train, eval(args.topN))
        if args.tst_w_val:
            test_results = evaluate(test_twv_loader, test_y_data, mask_tv, eval(args.topN))
        else:
            test_results = evaluate(test_loader, test_y_data, mask_tv, eval(args.topN))
        evaluate_utils.print_results(None, valid_results, test_results)

        if valid_results[1][1] > best_recall: # recall@20 as selection
            best_recall, best_epoch = valid_results[1][1], epoch
            best_results = valid_results
            best_test_results = test_results

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model, '{}{}_{}lr1_{}lr2_{}wd1_{}wd2_bs{}_cate{}_in{}_out{}_lam{}_dims{}_emb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_wmin{}_wmax{}_{}.pth' \
                .format(save_path, args.dataset, args.lr1, args.lr2, args.wd1, args.wd2, args.batch_size, args.n_cate, \
                args.in_dims, args.out_dims, args.lamda, args.mlp_dims, args.emb_size, args.mean_type, args.steps, \
                args.noise_scale, args.noise_min, args.noise_max, args.sampling_steps, args.reweight, args.w_min, args.w_max, args.log_name))
            torch.save(Autoencoder, '{}{}_{}lr1_{}lr2_{}wd1_{}wd2_bs{}_cate{}_in{}_out{}_lam{}_dims{}_emb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_wmin{}_wmax{}_{}_AE.pth' \
                .format(save_path, args.dataset, args.lr1, args.lr2, args.wd1, args.wd2, args.batch_size, args.n_cate, \
                args.in_dims, args.out_dims, args.lamda, args.mlp_dims, args.emb_size, args.mean_type, args.steps, \
                args.noise_scale, args.noise_min, args.noise_max, args.sampling_steps, args.reweight, args.w_min, args.w_max, args.log_name))
    
    print("Runing Epoch {:03d} ".format(epoch) + 'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime(
                        "%H: %M: %S", time.gmtime(time.time()-start_time)))
    print('---'*18)

print('==='*18)
print("End. Best Epoch {:03d} ".format(best_epoch))
evaluate_utils.print_results(None, best_results, best_test_results)   
print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))





