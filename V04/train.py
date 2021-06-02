import random

import argparse
import time


import numpy as np
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.nn as pyg_nn

import models
import utils
import matplotlib.pyplot as plt

def train(dataset, testset, task, args):

    cuda = torch.device('cuda')

    data_size = len(dataset)

    random.shuffle(dataset)

    loader = DataLoader(dataset[:int(data_size * 0.9)], batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(dataset[int(data_size * 0.9):], batch_size=args.batch_size, shuffle=True)

    model = models.GNNModule(args.num_feature, args.hidden_dim, 3, args, task=task)
    model = model.cuda()
    scheduler, opt = utils.build_optimizer(args, model.parameters())

    val = []


    for epoch in range(args.epochs):
        if epoch == 30:
            args.lr = args.lr2
            scheduler, opt = utils.build_optimizer(args, model.parameters())

        if epoch == 200:
            args.lr = args.lr3
            scheduler, opt = utils.build_optimizer(args, model.parameters())

        if epoch == 300:
            args.lr = args.lr4
            scheduler, opt = utils.build_optimizer(args, model.parameters())


        total_loss = 0
        model.train()
        for batch in loader:
            batch= batch.cuda()

            opt.zero_grad()
            pred = model(batch)
            label = batch.y
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        # print(total_loss)

        val_acc = test(validation_loader, model)
        val.append(val_acc)
        # if epoch % 10 == 0:
        # val_acc = test(validation_loader, model)
        print(epoch, total_loss, '  loss', val_acc, '%  valid')


    plt.plot(val)
    # plt.title("%s_%s" % (args.model_type, args.dataset))
    # plt.xlabel('Number of epochs')
    # plt.ylabel('Accuracy')
    # plt.savefig("%s_%s.png" % (args.model_type, args.dataset))

    test_loader = DataLoader(testset, batch_size=600)
    for data in test_loader:
        data = data.cuda()
        pred = model(data).max(dim=1)[1]
        # print(pred)
    pred_1 = pred.cpu() + np.ones(shape=(600))
    pred_2 = pred_1.numpy()
    pred_np = pred_2.astype(np.int64)
    f = open("result.csv","w")

    for i in range (600):
        f.write(str(pred_np[i]) + '\n')
    f.close()







def test(loader, model):
    model.eval()

    correct = 0
    for data in loader:
        data = data.cuda()
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            pred = model(data).max(dim=1)[1]
            label = data.y

        if model.task == 'node':
            mask = data.val_mask
            # node classification: only evaluate on nodes in test set
            pred = pred[mask]
            label = data.y[mask]

        correct += pred.eq(label).sum().item()

    if model.task == 'graph':
        total = len(loader.dataset)
    else:
        total = 0
        for data in loader.dataset:
            total += torch.sum(mask).item()
    return correct / total
