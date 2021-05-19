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
    data_size = len(dataset)

    random.shuffle(dataset)

    loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(dataset[int(data_size * 0.8):], batch_size=args.batch_size, shuffle=True)


    model = models.GNNModule(1, args.hidden_dim, 3, args, task=task)
    scheduler, opt = utils.build_optimizer(args, model.parameters())

    val = []
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in loader:
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
        print(val_acc, '  valid')

    plt.plot(val)
    # plt.title("%s_%s" % (args.model_type, args.dataset))
    # plt.xlabel('Number of epochs')
    # plt.ylabel('Accuracy')
    # plt.savefig("%s_%s.png" % (args.model_type, args.dataset))

    test_loader = DataLoader(testset, batch_size=600)
    for data in test_loader:
        pred = model(data).max(dim=1)[1]
        # print(pred)
    pred_1 = pred + np.ones(shape=(600))
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
