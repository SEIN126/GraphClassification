#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import matplotlib.pyplot as plt
import random
import time
import csv
import sys
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch import tensor
# from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader





from utils import *
# from models.gfn import ResGCN
# from models.gnn import GNNModule

random_seed = 77
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)


# In[2]:


def arg_parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, help='epochs', default=200)
    parser.add_argument('--train', type=str2bool, help='[True or False] train or not', default=False)
    parser.add_argument('--save_pt', type=str2bool, help='[True or False] save weight file or not', default=False)
    parser.add_argument('--save_fg', type=str2bool, help='[True or False] save val_acc, loss graph or not', default=False)
    parser.add_argument('--printed', type=str2bool, help='[True or False] print test file or not', default=False)

    return parser.parse_args()


# In[3]:


def str2bool(v): 
    if isinstance(v, bool): 
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'): return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'): return False 
    else: raise argparse.ArgumentTypeError('Boolean value expected.')


# In[4]:


def train_epochs(CONFIG, trainset, model, optimizer, scheduler, epochs, device, save=False, save_fg=False):
    '''
    StratifiedKFold
    When we made weight files, we have used StratifiedKFold cross validation.
    But in this code,
    we decide not to write it related down StratifiedKFold cross validation code because of its long long training time.
    so, if you want to those code using StratifiedKFold cross validation, please issue in our repo.
    thank you.
    '''

    best_acc = 0
    
    # random
    random.shuffle(trainset)

    train_dataset = trainset[:int(len(trainset)*CONFIG['train_val_ratio'])]
    val_dataset = trainset[int(len(trainset)*CONFIG['train_val_ratio']):]
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
    
    # start training
    val_losses, train_losses, train_accs, val_accs = [], [], [], []
    
    optimizer = optimizer
    scheduler = scheduler
    
    print('Train %s model'%(CONFIG['model']))
    for epoch in range(1, epochs + 1):
        # check time
        end = time.time()
        
        train_loss, train_acc = train(model, optimizer, scheduler, train_loader, device, CONFIG['weight_loss'])
        epoch_time = time.time()-end

        train_accs.append(train_acc)
        train_losses.append(train_loss)

        val_losses.append(eval_loss(model, val_loader, device, with_eval_mode=True))
        val_accs.append(eval_acc(model, val_loader, device, with_eval_mode=True))
        
        eval_info = {
            'epoch'      : epoch,
            'train_loss' : train_loss,
            'train_acc'  : train_acc,
            'val_acc'    : val_accs[-1],
            'val_loss'   : val_losses[-1],
            'time'       : epoch_time
        }
        
        
        if val_accs[-1] > best_acc:
            state = {
                'net'   : model.state_dict(),
                'acc'   : val_accs[-1],
                'epoch' : epoch
            }
            if save :
                print('saving...')
                timestamp = time.strftime('%m%d_%H%M', time.localtime())
                file_name = './ckpt/best_'+CONFIG['model']+timestamp+'_model.pt'
                torch.save(state, file_name)
            best_acc = val_accs[-1]

        if logger is not None:
            logger(eval_info)
        
    train_acc, val_acc = tensor(train_accs), tensor(val_accs)
    val_loss = tensor(val_losses)

    train_acc_mean = train_acc.mean().item()
    val_acc_mean = val_acc.mean().item()
    val_acc_std = val_acc.std().item()

    print('Train Acc: {:.4f}, Test Acc: {:.3f} ± {:.3f}'.format(train_acc_mean, val_acc_mean, val_acc_std))
    sys.stdout.flush()
    
    plot_graph(train_accs, val_accs, train_losses, val_loss, CONFIG['model'], save=save_fg)
    
    if epochs == 0 :
        # test case
        return 1, 1
    
    return best_acc, val_loss[state['epoch']-1]
    


# In[12]:


def train(model, optimizer, scheduler, loader, device, weighted_loss):
    model.train()

    total_loss = 0
    correct = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        
        loss = F.nll_loss(out, data.y.view(-1))
        if str2bool(weighted_loss) :
            '''
            When training the model, consider the number of samples in the class 
            and design the loss function by applying weights.
            '''
            w = torch.tensor([1.0,2.0,1.5]).to(device)
            loss = F.nll_loss(out, data.y.view(-1), weight=w)
        
        pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
        
        # scheduler update
        if scheduler is not None:
            scheduler.step()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


# In[13]:


def eval_acc(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)


# In[14]:


def main(epochs, training=False, save_pt=False, save_fg=False, printed=False):
    # device setting
    print('train : ', train)
    print('save the ckpt : ', save_pt)
    print('save the figure : ', save_fg)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('===>', device)
    
    GCN_CONFIG, GFN1_CONFIG, GFN2_CONFIG, GAT_CONFIG, GS_CONFIG = model_config()
    
    gcn_trainset, gcn_testset = get_dataset(GCN_CONFIG)
    gfn_trainset, gfn_testset = get_dataset(GFN1_CONFIG)
    gat_trainset, gat_testset = get_dataset(GAT_CONFIG)
    gs_trainset, gs_testset  = get_dataset(GS_CONFIG)
    
    gcn_model, gfn_model1, gfn_model2, gat_model, gs_model = get_model(gfn_trainset)
    print('===> model ready')
    
    # model to gpu, get scheduler, optimizer
    gcn_model.to(device)
    gfn_model1.to(device)
    gfn_model2.to(device)
    gat_model.to(device)
    gs_model.to(device)
    
    if training :
        print()
        print('===> model train')
        scheduler_gcn, optimizer_gcn   = build_optimizer(GCN_CONFIG, gcn_model.parameters())
        scheduler_gfn1, optimizer_gfn1 = build_optimizer(GFN1_CONFIG, gfn_model1.parameters())
        scheduler_gfn2, optimizer_gfn2 = build_optimizer(GFN2_CONFIG, gfn_model2.parameters())
        scheduler_gat, optimizer_gat   = build_optimizer(GAT_CONFIG, gat_model.parameters())    
        scheduler_gs, optimizer_gs     = build_optimizer(GS_CONFIG, gs_model.parameters())


        # gcn training
        gcn_val_acc, gcn_val_loss = train_epochs(GCN_CONFIG, 
                                             gcn_trainset, 
                                             gcn_model, 
                                             optimizer_gcn, 
                                             scheduler_gcn, 
                                             epochs,
                                             device,
                                             save=save_pt,
                                             save_fg=save_fg
                                             )

        # gfn1 training
        gfn_val_acc, gfn_val_loss = train_epochs(GFN1_CONFIG, 
                                             gfn_trainset, 
                                             gfn_model1, 
                                             optimizer_gfn1, 
                                             scheduler_gfn1, 
                                             epochs,
                                             device,
                                             save=save_pt,
                                             save_fg=save_fg
                                             )
        # gfn2 training - with loss weight
        gfn_val_acc, gfn_val_loss = train_epochs(GFN2_CONFIG, 
                                             gfn_trainset, 
                                             gfn_model2, 
                                             optimizer_gfn2, 
                                             scheduler_gfn2, 
                                             epochs,
                                             device,
                                             save=save_pt,
                                             save_fg=save_fg
                                             )

        # gat training
        gat_val_acc, gat_val_loss = train_epochs(GAT_CONFIG, 
                                             gat_trainset, 
                                             gat_model, 
                                             optimizer_gat, 
                                             scheduler_gat, 
                                             epochs,
                                             device,
                                             save=save_pt,
                                             save_fg=save_fg
                                             )

        # gs training
        gs_val_acc, gs_val_loss = train_epochs(GS_CONFIG, 
                                             gs_trainset, 
                                             gs_model, 
                                             optimizer_gs, 
                                             scheduler_gs, 
                                             epochs,
                                             device,
                                             save=save_pt,
                                             save_fg=save_fg
                                             )
        print('===> model train complete')
        
    # make test loader
    loader_gcn = DataLoader(gcn_testset, len(gcn_testset), shuffle=False)
    loader_gfn = DataLoader(gfn_testset, len(gfn_testset), shuffle=False)
    loader_gat = DataLoader(gat_testset, len(gat_testset), shuffle=False)
    loader_gs  = DataLoader(gs_testset, len(gs_testset), shuffle=False)
    
    loaders = [loader_gcn, loader_gfn, loader_gfn, loader_gat, loader_gs]
    models = [gcn_model, gfn_model1, gfn_model2, gat_model, gs_model]
    
    # make test result
    result = test_ensembled(models, loaders, device, printed=False)
    print('===> model test complete')

    # make test file
    write_result(result)
    print('===> check the test file!')
    


# In[20]:


def test_ensembled(models, loaders, device, printed=False):
    '''
    inference with Ensemble(hard voting)
    '''
    num_class = 3
    gcn_model, gfn_model1, gfn_model2, gat_model, gs_model = models[0], models[1], models[2], models[3], models[4] 
    if models[0] is not None:
        ck_gcn = torch.load('./ckpt/best_GCN_model.pt', map_location=device)
        gcn_model.load_state_dict(ck_gcn['net'])
    if models[1] is not None:
        ck_gfn1 = torch.load("./ckpt/best_GFN1_model.pt", map_location=device)
        gfn_model1.load_state_dict(ck_gfn1['net'])
    if models[2] is not None:
        ck_gfn2 = torch.load("./ckpt/best_GFN2_model.pt", map_location=device)
        gfn_model2.load_state_dict(ck_gfn2['net'])            
    if models[3] is not None:
        ck_gat = torch.load('./ckpt/best_GAT_model.pt', map_location=device)
        gat_model.load_state_dict(ck_gat['net'])
    if models[4] is not None:
        ck_gs = torch.load('./ckpt/best_GraphSage_model.pt', map_location=device)
        gs_model.load_state_dict(ck_gs['net'])

    models = [gcn_model, gfn_model1, gfn_model2, gat_model, gs_model]

    preds = list()
    for model, loader in zip(models, loaders):
        if model is not None and loaders is not None :
            model.cpu()
            model.eval()
            
            pred = list()
            for data in loader:
                with torch.no_grad():
                    pred += model(data).max(1)[1]
            preds.append(pred)
    
    result = list()      
    
    for i in range(600):
        votes = torch.tensor([preds[0][i], preds[1][i], preds[2][i], preds[3][i], preds[4][i]])
        
        if printed :
            print(votes)
        
        tmp = torch.zeros((len(votes), num_class))
        tmp[range(len(votes)),votes] = 1
        vals, nums = torch.topk(torch.sum(tmp, 0), 2)
        if vals[0] == vals[1]:
            if printed :
                print("vals:", int(vals[0].item()), int(vals[1].item()), "nums:", nums[0].item(), nums[1].item())
                #v = nums[np.random.randint(2)]
                #print("randomly picked: ", v)
                print("gfn predicted class:", votes[1].item())
                print('---------------------------------')
            result.append(votes[1])
            
        else:
            result.append(nums[0])
            if printed :
                print("absolutely picked:", nums[0].item())
                print('---------------------------------')

    return result
    


# In[21]:


if __name__ == '__main__':
    args = arg_parse()
    
    epochs   = args.epochs
    training = args.train
    save_pt  = args.save_pt
    save_fg  = args.save_fg
    printed  = args.printed
    print(training, save_pt, save_fg,printed)
    main(epochs, training=training, save_pt=save_pt, save_fg=save_fg, printed=printed)





