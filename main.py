import dataset_generation
import train
from torch_geometric.datasets import TUDataset
import torch_geometric.datasets
import torch_geometric.data
import torch.utils.data

class args:
    model_type = 'GraphSage'
    num_layers = 10
    batch_size = 100
    hidden_dim = 10
    dropout = 0.2
    epochs = 10
    opt = 'adam'  # opt_parser
    opt_scheduler = 'none'
    weight_decay = 0.01
    lr = 0.01

dataset_train, dataset_test = dataset_generation.load()

task = 'graph'
train.train(dataset_train, dataset_test, task, args)

0;


