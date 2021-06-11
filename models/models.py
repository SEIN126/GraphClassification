import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

class GNNModule(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args, task='node'):
        super(GNNModule, self).__init__()
        
        conv_model = self.build_conv_model(args.model_type)
        
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        
        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        
        for l in range(args.num_layers-1):
            self.convs.append(conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.Dropout(args.dropout), 
            nn.Linear(hidden_dim, output_dim))

        self.task = task
        if not self.task == 'graph':
            raise RuntimeError('Unknown task.')

        self.dropout = args.dropout
        self.num_layers = args.num_layers

    def build_conv_model(self, model_type):
        if model_type == 'GCN':
            return pyg_nn.GCNConv

        elif model_type == 'GAT':
            return GAT
        
        elif model_type == 'GCNtoGAT':
            return pyg_nn.GCNConv, GAT

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Each layer in GNN should consist of a convolution (specified in model_type),
        # a non-linearity , dropout. 

        for layer in self.convs :
            x = layer(x, edge_index)
        x = nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        if self.task == 'graph' :
            
            x = pyg_nn.global_max_pool(x, batch)

        x = self.post_mp(x)

        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class GraphSage(pyg_nn.MessagePassing):
    """Non-minibatch version of GraphSage."""
    def __init__(self, in_channels, out_channels, reducer='mean', 
                 normalize_embedding=True):
        super(GraphSage, self).__init__(aggr='mean')

       
        # Define the layers needed for the message function. 
        self.lin = nn.Linear(in_channels+out_channels, out_channels, bias = False) # TODO
        self.agg_lin = nn.Linear(in_channels, out_channels) # TODO



        if normalize_embedding:
            self.normalize_emb = True

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, size=(num_nodes, num_nodes), x=x)

    def message(self, x_j):
        
        # Compute message from x_j to x_i
        # RELU non-linearity, and a mean aggregator.
        x_j = self.agg_lin(x_j)
        x_j = F.relu(x_j)

        return x_j

    def update(self, aggr_out, x):
        
        self.lin(torch.cat((x,aggr_out),dim = 1))
        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out)

        return aggr_out


class GAT(pyg_nn.MessagePassing):

    def __init__(self, in_channels, out_channels, num_heads=1, concat=True,
                 dropout=0, bias=True, **kwargs):
        super(GAT, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = num_heads
        self.concat = concat 
        self.dropout = dropout

    
        # Define the layers needed for the forward function. 
        self.lin = nn.Linear(in_channels , out_channels, bias = False) # TODO

        self.att = nn.Parameter(torch.Tensor(out_channels*2,1)) # TODO

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(self.heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

        

    def forward(self, x, edge_index, size=None):
        x = self.lin(x) # TODO
        

        # Start propagating messages.
        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        #  Constructs messages to node i for each edge (j, i)
        e = F.leaky_relu(torch.cat((x_i,x_j),dim = 1).mm(self.att), negative_slope = 0.2) # (nodes, 2*f) * (2*f, 1) => (nodes, 1), neagative slope 0.2 
        alpha = pyg_utils.softmax(e,edge_index_i) # TODO
        x_j = alpha*x_j

        return x_j

    def update(self, aggr_out):
        # Updates node embedings.
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out
