from argparse import ArgumentParser

from initialisation import init_qz
from datasets import IHDP
from evaluation import Evaluator, get_y0_y1
from networks_ks import p_x_z, p_t_z, p_y_zt, q_t_x, q_y_xt, q_z_tyx

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
from torch.distributions import normal
from torch import optim

from samplers import gen_KS_samples
from sklearn.model_selection import train_test_split

import pickle
import os

parser = ArgumentParser()

# Set Hyperparameters
parser.add_argument('-N', type=int, default=2000)
parser.add_argument('-reps', type=int, default=1)
parser.add_argument('-z_dim', type=int, default=10)
parser.add_argument('-h_dim', type=int, default=16)
parser.add_argument('-epochs', type=int, default=200)
parser.add_argument('-batch', type=int, default=100)
#parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-lr', type=float, default=0.0001)
parser.add_argument('-decay', type=float, default=0.01)
parser.add_argument('-print_every', type=int, default=10)

args = parser.parse_args()

directory = os.getcwd()+'/clean_out/'
if not os.path.exists(directory):
    os.makedirs(directory) 

result = []
for seed in np.arange(1,13):
    try:
        (X, Y, T) = gen_KS_samples(args.N, seed=seed)
        
        idxtrain, idxte = train_test_split(np.arange(X.shape[0]), test_size=0.2, random_state=1)
        
        (Xtr, Ytr0, Ttr) = (X[idxtrain], Y[idxtrain,None], T[idxtrain,None])
        (Xte, Yte0, Tte) = (X[idxte], Y[idxte,None], T[idxte,None])
        
        # zero mean, unit variance for y during training, use ym & ys to correct when using testset
        ym, ys = np.mean(Ytr0), np.std(Ytr0)
        Ytr, Yte = (Ytr0 - ym) / ys, (Yte0 - ym) / ys
        
        
        # init networks (overwritten)
        x_dim = X.shape[-1]
        
        p_x_z_dist = p_x_z(dim_in=args.z_dim, nh=2, dim_h=args.h_dim, dim_out_con=x_dim) 
        p_t_z_dist = p_t_z(dim_in=args.z_dim, nh=1, dim_h=args.h_dim, dim_out=1) 
        p_y_zt_dist = p_y_zt(dim_in=args.z_dim, nh=2, dim_h=args.h_dim, dim_out=1) 
        
        p_z_dist = normal.Normal(torch.zeros(args.z_dim) , torch.ones(args.z_dim))
        
        q_t_x_dist = q_t_x(dim_in=x_dim, nh=1, dim_h=args.h_dim, dim_out=1) 
        # t is not feed into network, therefore not increasing input size (y is fed).
        q_y_xt_dist = q_y_xt(dim_in=x_dim, nh=2, dim_h=args.h_dim, dim_out=1) 
        q_z_tyx_dist = q_z_tyx(dim_in=x_dim + 1, nh=2, dim_h=args.h_dim,
                               dim_out=args.z_dim) 
        
        
        # Create optimizer
        params = list(p_x_z_dist.parameters()) + \
                 list(p_t_z_dist.parameters()) + \
                 list(p_y_zt_dist.parameters()) + \
                 list(q_t_x_dist.parameters()) + \
                 list(q_y_xt_dist.parameters()) + \
                 list(q_z_tyx_dist.parameters())
        
        # Adam is used, like original implementation, in paper Adamax is suggested
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
        
        # init q_z inference
        q_z_tyx_dist = init_qz(q_z_tyx_dist, p_z_dist, Ytr, Ttr, Xtr)
        
        M = args.batch
        n_epoch, n_iter_per_epoch, idx = args.epochs, 10 * int(Xtr.shape[0] / M), list(range(Xtr.shape[0]))
        
        loss = defaultdict(list)
        
        rr = []
        for epoch in range(n_epoch):
            # print('Epoch: %i/%i' % (epoch, n_epoch))
            if epoch < 150:
                scheduler.step()
            loss_sum = 0.
            # shuffle index
            np.random.shuffle(idx)
            # take random batch for training
            for j in range(n_iter_per_epoch):
                batch = np.random.choice(idx, M)
                x_train, y_train, t_train = torch.FloatTensor(Xtr[batch]), torch.FloatTensor(Ytr[batch]), \
                                            torch.FloatTensor(Ttr[batch])
                # inferred distribution over z
                xy = torch.cat((x_train, y_train), 1)
                z_infer = q_z_tyx_dist(xy=xy, t=t_train)
                # use a single sample to approximate expectation in lowerbound
                z_infer_sample = z_infer.sample()
                
                # RECONSTRUCTION LOSS
                # p(x|z)
                x_con = p_x_z_dist(z_infer_sample)
                l2 = x_con.log_prob(x_train).sum(1)
                loss['Reconstr_x_con'].append(l2.sum().cpu().detach().float())
                # p(t|z)
                t = p_t_z_dist(z_infer_sample)
                l3 = t.log_prob(t_train).squeeze()
                loss['Reconstr_t'].append(l3.sum().cpu().detach().float())
                # p(y|t,z)
                # for training use t_train, in out-of-sample prediction this becomes t_infer
                y = p_y_zt_dist(z_infer_sample, t_train)
                l4 = y.log_prob(y_train).squeeze()
                loss['Reconstr_y'].append(l4.sum().cpu().detach().float())
                
                # REGULARIZATION LOSS
                # p(z) - q(z|x,t,y)
                # approximate KL
                l5 = (p_z_dist.log_prob(z_infer_sample) - z_infer.log_prob(z_infer_sample)).sum(1)
                loss['Regularization'].append(l5.sum().cpu().detach().float())
        
                # AUXILIARY LOSS
                # q(t|x)
                t_infer = q_t_x_dist(x_train)
                l6 = t_infer.log_prob(t_train).squeeze()
                loss['Auxiliary_t'].append(l6.sum().cpu().detach().float())
                # q(y|x,t)
                y_infer = q_y_xt_dist(x_train, t_train)
                l7 = y_infer.log_prob(y_train).squeeze()
                loss['Auxiliary_y'].append(l7.sum().cpu().detach().float())
                
                # Total objective
                # inner sum to calculate loss per item, torch.mean over batch
                loss_mean = torch.mean(l2 + l3 + l4 + l5 + l6 + l7)
                loss['Total'].append(loss_mean.cpu().detach().numpy())
                objective = -loss_mean
        
                optimizer.zero_grad()
                # Calculate gradients
                objective.backward()
                # Update step
                optimizer.step()
                
            if epoch % args.print_every == 0:
                print('Epoch %i' % epoch)
                print(objective)
                # y0, y1 = get_y0_y1(p_y_zt_dist, q_y_xt_dist, q_z_tyx_dist, torch.tensor(Xte),
                #                     torch.tensor(Tte))
                # y0, y1 = y0 * ys + ym, y1 * ys + ym
                # print(np.mean(y1)-np.mean(y0))
            
            # if epoch>(n_epoch-20):
            #     y0, y1 = get_y0_y1(p_y_zt_dist, q_y_xt_dist, q_z_tyx_dist, torch.tensor(Xte),
            #                         torch.tensor(Tte))
            #     y0, y1 = y0 * ys + ym, y1 * ys + ym
            #     rr.append(np.mean(y1)-np.mean(y0))
        rr = []
        for j in range(500):
            y0, y1 = get_y0_y1(p_y_zt_dist, q_y_xt_dist, q_z_tyx_dist, torch.tensor(Xte),
                                      torch.tensor(Tte))
            y0, y1 = y0 * ys + ym, y1 * ys + ym
            rr.append(np.mean(y1)-np.mean(y0))
            # if epoch>(n_epoch-20):
            #     for j in range(200):
            #         y0, y1 = get_y0_y1(p_y_zt_dist, q_y_xt_dist, q_z_tyx_dist, torch.tensor(Xte),
            #                                   torch.tensor(Tte))
            #         y0, y1 = y0 * ys + ym, y1 * ys + ym
            #         rr.append(np.mean(y1)-np.mean(y0))
                
        print('####',seed)
        print(np.mean(rr))
        print('####')
        result.append((seed, np.mean(rr)))
        pickle.dump(result, open(directory+'clean5', 'wb')) 
    except:
        pass
    
    
    
    
    
    









