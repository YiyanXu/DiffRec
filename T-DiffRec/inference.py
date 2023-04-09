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

import models.gaussian_diffusion as gd
from models.DNN import DNN
import evaluate_utils
import data_utils
from copy import deepcopy

import random
random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) # gpu
np.random.seed(random_seed) # numpy
random.seed(random_seed) # random and transforms
torch.backends.cudnn.deterministic=True # cudnn
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='yelp_clean', help='choose the dataset')
parser.add_argument('--data_path', type=str, default='./datasets/', help='load data path')
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
parser.add_argument('--log_name', type=str, default='log', help='the log name')

parser.add_argument('--w_min', type=float, default=0.1, help='the minimum weight for interactions')
parser.add_argument('--w_max', type=float, default=1., help='the maximum weight for interactions')

# params for diffusion
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=5, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')

args = parser.parse_args()

args.data_path = args.data_path + args.dataset + '/'
if args.dataset == 'amazon-book_clean':
    args.steps = 10
    args.noise_scale = 0.0005
    args.noise_min = 0.001
    args.noise_max = 0.005
    args.w_min = 0.1
    args.w_max = 1.0
elif args.dataset == 'yelp_clean':
    args.steps = 5
    args.noise_scale = 0.005
    args.noise_min = 0.001
    args.noise_max = 0.01
    args.w_min = 0.5
    args.w_max = 1.0
else:
    raise ValueError

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


### CREATE DIFFUISON ###
if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError("Unimplemented mean type %s" % args.mean_type)

diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, \
        args.noise_scale, args.noise_min, args.noise_max, args.steps, device)
diffusion.to(device)

### CREATE DNN ###
model_path = "../checkpoints/T-DiffRec/"
if args.dataset == "amazon-book_clean":
    model_name = "amazon-book_clean_lr1e-05_wd0.0_bs400_dims[1000]_emb10_x0_steps10_scale0.0005_min0.001_max0.005_sample0_reweight1_wmin0.1_wmax1.0_log.pth"
elif args.dataset == "yelp_clean":
    model_name = "yelp_clean_lr1e-05_wd0.0_bs400_dims[1000]_emb10_x0_steps5_scale0.005_min0.001_max0.01_sample0_reweight1_wmin0.5_wmax1.0_log.pth"

model = torch.load(model_path + model_name).to(device)

print("models ready.")

def evaluate(data_loader, data_te, mask_his, topN):
    model.eval()
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]

    predict_items = []
    target_items = []
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]]
            batch = batch.to(device)
            prediction = diffusion.p_sample(model, batch, args.sampling_steps, args.sampling_noise)
            prediction[his_data.nonzero()] = -np.inf

            _, indices = torch.topk(prediction, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)

    test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)

    return test_results

valid_results = evaluate(test_loader, valid_y_data, train_data, eval(args.topN))
if args.tst_w_val:
    test_results = evaluate(test_twv_loader, test_y_data, mask_tv, eval(args.topN))
else:
    test_results = evaluate(test_loader, test_y_data, mask_tv, eval(args.topN))
evaluate_utils.print_results(None, valid_results, test_results)






