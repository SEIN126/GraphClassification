import dataset_generation
import train
import random
from torch_geometric.datasets import TUDataset
import torch_geometric.datasets
import torch_geometric.data
import torch.utils.data

class args:
    model_type = 'GCN'
    num_layers = 7
    batch_size = 100
    hidden_dim = [10,20,30,40,30,20,10]
    dropout = 0.1
    epochs = 50
    opt = 'adam'  # opt_parser
    momentum = 0.95
    opt_scheduler = 'none'
    weight_decay = 0.01
    lr = 1e-2
    lr2 = 1e-3
    lr3 = 1e-4
    lr4 = 1e-5
    T_max = 5
    num_feature = 6

dataset_train, dataset_test = dataset_generation.load()
dataset_train_0 = []
dataset_train_1 = []
dataset_train_2 = []

for i in range(len(dataset_train)):
    if dataset_train[i].y == 0: # 2400개
        dataset_train_0.append(dataset_train[i])
    elif dataset_train[i].y == 1: # 575개
        dataset_train_1.append(dataset_train[i])
    elif dataset_train[i].y == 2: # 1425개
        dataset_train_2.append(dataset_train[i])

random.shuffle(dataset_train_0)
random.shuffle(dataset_train_1)
random.shuffle(dataset_train_2)

dataset_fair = []
# range 575 로 하고 그냥 i를 넣으면 최소 길이에 맞춘 데이;터
# range 2400으로 하고 나머지 들어가면 최대 길이에 맞춰서 중복 허용
# for i in range(2400):
for i in range(575):
    # dataset_fair.append(dataset_train_0[i])
    # dataset_fair.append(dataset_train_1[i%575])
    # dataset_fair.append(dataset_train_2[i%1425])
    dataset_fair.append(dataset_train_0[i])
    dataset_fair.append(dataset_train_1[i])
    dataset_fair.append(dataset_train_2[i])
random.shuffle(dataset_fair)



task = 'graph'
train.train(dataset_fair, dataset_test, task, args)

0;


