# -*- coding: utf-8 -*-
# +
import os
import re
import networkx as nx
import torch
from torch_geometric.data import Data, Dataset
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import csv


import sys
import time
import torch
import torch.nn.functional as F
from torch import tensor
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from functools import partial
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import global_mean_pool, global_add_pool
from models.gfn import ResGCN
from models.gnn import GNNModule
import torch.optim as optim

def write_result(result):
    # read the test file
    root = './data'
    test_graphs = list()
    test_path = os.path.join('./data', 'test.txt')

    with open(test_path) as f:
        for row in f.readlines():
            test_graphs.append(int(row.rstrip())-1)
    
    timestamp = time.strftime('%Y%m%d_%H%M', time.localtime())
    outfile = open(os.path.join(root, 'result_'+timestamp+'.csv'), 'w', newline='')
    writer = csv.writer(outfile)
    for i in range(len(test_graphs)):
        if i == 0:
            writer.writerow(('Id', 'Category'))
        writer.writerow((test_graphs[i]+1, result[i].item()+1))
    outfile.close()

def build_optimizer(CONFIG, params):
    filter_fn = filter(lambda p : p.requires_grad, params)
    if CONFIG['opt'] == 'adam':
        optimizer = optim.Adam(filter_fn, lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    elif CONFIG['opt'] == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=CONFIG['lr'], momentum=0.9, weight_decay=CONFIG['weight_decay'], nesterov=True)

    if CONFIG['opt_scheduler'] == 'none':
        return None, optimizer
    elif CONFIG['opt_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5, eta_min=1e-5)
    elif CONFIG['opt_scheduler'] == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=5e-4)
    return scheduler, optimizer

def model_config():
    GCN_CONFIG = {
        'model'           : 'GCN',
        'data'            : 'nx6',
        'd_type'          : 'oversampling', 
        'train_val_ratio' : 0.9,
        'batch_size'      : 100,
        'opt'             : 'adam',
        'lr'              : 5e-3, 
        'hidden_dim'      : 30,
        'weight_decay'    : 0.01,
        'opt_scheduler'   : 'cos',
        'T_max'           : 5,
        'dropout'         : 0.1,
        'num_layers'      : 5,
        'num_feature'     : 6,
        'weight_loss'     : 'true'
    }
    
    GFN1_CONFIG = {
        'model'           : 'GFN',    
        'data'            : 'nx9',
        'd_type'          : 'oversampling',
        'train_val_ratio' : 0.8,
        'batch_size'      : 128,
        'opt'             : 'sgd',
        'lr'              : 0.001,
        'weight_decay'    : 5e-3,
        'opt_scheduler'   : 'cos',
        'dropout'         : 0,
        'edge_norm'       : False,
        'hidden'          : 128,
        'num_feat_layers' : 1,
        'num_conv_layers' : 3,
        'num_fc_layers'   : 2,
        'residual'        : True,
        'skip_connection' : True,
        'edge_norm'       : False,
        'res_branch'      : 'BNConvReLU',
        'global_pool'     : 'sum',
        'num_features'    : 9,
        'num_classes'     : 3,
        'weight_loss'     : 'false'
    }
    
    GFN2_CONFIG = {
        'model'           : 'GFN',    
        'data'            : 'nx9',
        'd_type'          : 'oversampling',
        'train_val_ratio' : 0.8,
        'batch_size'      : 128,
        'opt'             : 'sgd',
        'lr'              : 0.001,
        'weight_decay'    : 5e-3,
        'opt_scheduler'   : 'cos',
        'dropout'         : 0,
        'edge_norm'       : False,
        'hidden'          : 128,
        'num_feat_layers' : 1,
        'num_conv_layers' : 3,
        'num_fc_layers'   : 2,
        'residual'        : True,
        'skip_connection' : True,
        'edge_norm'       : False,
        'res_branch'      : 'BNConvReLU',
        'global_pool'     : 'sum',
        'num_features'    : 9,
        'num_classes'     : 3,
        'weight_loss'     : 'true'
    }

    GAT_CONFIG = {
        'model'           : 'GAT',
        'data'            : 'nx9',
        'd_type'          : 'oversampling', 
        'train_val_ratio' : 0.8,
        'batch_size'      : 128,
        'opt'             : 'sgd',
        'lr'              : 0.001,  
        'hidden_dim'      : 32,
        'weight_decay'    : 0.0, 
        'opt_scheduler'   : 'cos',
        'dropout'         : 0.2,
        'num_layers'      : 3,
        'num_feature'     : 9,
        'weight_loss'     : 'true'    
    }

    GS_CONFIG = {
        'model'           : 'GraphSage',
        'data'            : 'nx9',
        'd_type'          : 'oversampling', 
        'train_val_ratio' : 0.8,
        'batch_size'      : 128,
        'opt'             : 'adam',
        'lr'              : 0.001,  
        'hidden_dim'      : 32,
        'weight_decay'    : 5e-3, 
        'opt_scheduler'   : 'cos',
        'dropout'         : 0.0,
        'num_layers'      : 3,
        'num_feature'     : 9,
        'weight_loss'     : 'true' 
    }
    
    return GCN_CONFIG, GFN1_CONFIG, GFN2_CONFIG, GAT_CONFIG, GS_CONFIG

def get_model(dataset):
    GCN_CONFIG, GFN1_CONFIG, GFN2_CONFIG, GAT_CONFIG, GS_CONFIG = model_config()
    
    
    
    gcn_model = GNNModule(GCN_CONFIG['num_feature'], GCN_CONFIG['hidden_dim'], 3, GCN_CONFIG, task='graph')
    
    gfn_model1 = ResGCN(dataset, 
                       GFN1_CONFIG["num_features"], 
                       GFN1_CONFIG['num_classes'], 
                       GFN1_CONFIG['hidden'], 
                       GFN1_CONFIG['num_feat_layers'], 
                       GFN1_CONFIG['num_conv_layers'],
                       GFN1_CONFIG['num_fc_layers'], 
                       gfn=True, collapse=False,
                       residual=GFN1_CONFIG['residual'], 
                       res_branch=GFN1_CONFIG['res_branch'],                      
                       global_pool=GFN1_CONFIG['global_pool'], 
                       dropout=GFN1_CONFIG['dropout'],
                       edge_norm=GFN1_CONFIG['edge_norm'])
    
    gfn_model2 = ResGCN(dataset, 
                       GFN2_CONFIG["num_features"], 
                       GFN2_CONFIG['num_classes'], 
                       GFN2_CONFIG['hidden'], 
                       GFN2_CONFIG['num_feat_layers'], 
                       GFN2_CONFIG['num_conv_layers'],
                       GFN2_CONFIG['num_fc_layers'], 
                       gfn=True, collapse=False,
                       residual=GFN2_CONFIG['residual'], 
                       res_branch=GFN2_CONFIG['res_branch'],                      
                       global_pool=GFN2_CONFIG['global_pool'], 
                       dropout=GFN2_CONFIG['dropout'],
                       edge_norm=GFN2_CONFIG['edge_norm'])
    
    gat_model = GNNModule(GAT_CONFIG['num_feature'], GAT_CONFIG['hidden_dim'], 3, GAT_CONFIG, task='graph')
    gs_model  = GNNModule(GS_CONFIG['num_feature'], GS_CONFIG['hidden_dim'], 3, GS_CONFIG, task='graph')

    return gcn_model, gfn_model1, gfn_model2, gat_model, gs_model

def get_dataset(CONFIG):

    '''
    1. choose the dataset
     - origin : original dataset
     - nx6    : # of feature 6 ; []
     - nx8    : # of feature 8 ; []
     - nx9    : # of feature 9 ; []

    => default : nx9

    2. choose the data sampling type
     - origin        : original data
     - undersampling : get randomly from the class 1, 3 so that each class has the same proportion 
     - oversampling  : get duplicate data so that each class has the smae proportion
    '''

    root = "./data"
    data = CONFIG['data']+'.pt'
    test_data = CONFIG['data']+'_test'+'.pt'

    trainset = torch.load(os.path.join(root, data))
    testset = torch.load(os.path.join(root, test_data))

    # choose the dataset type
    if CONFIG['d_type'] == 'undersampling':
        train_dataset = make_undersample_dataset(trainset)
        return train_dataset, testset
        
    elif CONFIG['d_type'] == 'oversampling':
        train_dataset = make_oversample_dataset(trainset)
        return train_dataset, testset
    
    elif CONFIG['d_type'] == 'original':
        return trainset, testset
    
def plot_graph(train_accs, test_accs, train_loss, val_loss, name, save=False):
    fig, ax = plt.subplots(1,2, figsize=(18,5))
    
    ax[0].plot(train_accs, label = 'train accuracy')
    ax[0].plot(test_accs, label = 'val accuracy')
    ax[0].set_title("accuracy")
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('accuracy')
    ax[0].legend(loc='upper left')
    
    ax[1].plot(train_loss, label = 'train_loss')
    ax[1].plot(val_loss, label = 'val_loss')
    ax[1].set_title("loss")
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend(loc='upper left')
    
    plt.show()
    if save :
        plt.savefig(name)


def print_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
            
def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)
    
def logger(info):
    epoch = info['epoch']
    if epoch == 1 or epoch % 10 == 0:
        train_acc, train_loss, val_acc, val_loss, time = info['train_acc'], info['train_loss'], info['val_acc'], info['val_loss'], info['time']
        print('epoch: {:03d}  |  Train Acc: {:.3f}  |  Train Loss: {:.3f}  |  Val Accuracy: {:.3f}  |  Val Loss: {:.3f}  |  Time: {:.3f}'.format(
            epoch, train_acc, train_loss, val_acc, val_loss, time))
        
def make_oversample_dataset(train_data):
    # 각 class의 갯수를 2400 근처로 맞춰줌 
    l1_data, l2_data, l3_data = [], [], []
    for data in train_data:
        if data.y.item() == 0:
            l1_data.append(data)
        if data.y.item() == 1:
            l2_data.append(data)
        if data.y.item() == 2:
            l3_data.append(data)
    
    l1_len = len(l1_data)
    l2_len = len(l2_data)
    l3_len = len(l3_data)
    max_len = max(max(l1_len,l2_len),l3_len)
    oversample_data = []
    
    # l1_data
    for i in range(round(max_len/l1_len)):
        oversample_data.extend(l1_data)
    # l2_data
    for i in range(round(max_len/l2_len)):
        oversample_data.extend(l2_data)
    # l3_data
    for i in range(round(max_len/l3_len)):
        oversample_data.extend(l3_data)
    return oversample_data
    
def make_undersample_dataset(train_data):
    # 샘플끼리의 비율을 정확히 같도록.
    l1_data, l2_data, l3_data = [], [], []
    for data in train_data:
        if data.y.item() == 0:
            l1_data.append(data)
        if data.y.item() == 1:
            l2_data.append(data)
        if data.y.item() == 2:
            l3_data.append(data)
    
    random.shuffle(l1_data) 
    random.shuffle(l2_data)
    random.shuffle(l3_data)
    
    undersample_data = []
    undersample_data.extend(l1_data[:575])
    undersample_data.extend(l2_data[:575])
    undersample_data.extend(l3_data[:575])
    return undersample_data
