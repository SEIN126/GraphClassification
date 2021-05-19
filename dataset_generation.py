import torch
import numpy as np
from torch_geometric.data import Data

def generation():
    file_graph_ind = './data/graph_ind.txt'
    with open(file_graph_ind) as data:
        lines_graph_ind = data.readlines()
    graph_ind = []
    for line in lines_graph_ind:
        line_stripped = line.rstrip()
        graph_ind.append([int(i) for i in line_stripped.split(', ')])

    file_graph = './data/graph.txt'
    with open(file_graph) as data:
        lines_graph = data.readlines()
    graph = []
    for line in lines_graph:
        line_stripped = line.rstrip()
        graph.append([int(i) for i in line_stripped.split(', ')])

    file_train = './data/train.txt'
    with open(file_train) as data:
        lines_train = data.readlines()
    train = []
    for line in lines_train:
        line_stripped = line.rstrip()
        train.append([int(i) for i in line_stripped.split(' ')])

    file_test = './data/test.txt'
    with open(file_test) as data:
        lines_test = data.readlines()
    test = []
    for line in lines_test:
        line_stripped = line.rstrip()
        test.append([int(line_stripped)])
    # 데이터 읽어와서 list로 재배열

    nodes = []
    newlist = []
    no_graph = 1
    for i in range(len(graph_ind)):
        if graph_ind[i][1] == no_graph:
            newlist.append(graph_ind[i][0])
        else:
            no_graph = no_graph + 1
            nodes.append(newlist)
            newlist = []
            newlist.append(graph_ind[i][0])

    nodes.append(newlist)
    # Graph_ind 의 번호에 맞추어 각 row에 해당 그래프의 노드들 삽입

    no_graph = 0
    node_idx = nodes[no_graph][:]
    node_set = set(node_idx)
    edges = []
    dataset = []
    for i in range(len(graph)):
        if i % 5000000 == 0:
            print(i)
        edge_set = set(graph[i][:])
        if edge_set.issubset(node_set):
            edges.append(graph[i][:])

        else:
            features = torch.tensor(np.random(shape=(len(node_idx), 1)), dtype=torch.float)
            # features = torch.tensor(np.random.rand(len(node_idx), 1), dtype=torch.float)

            edges_tensor = torch.tensor(edges-np.ones(shape=(len(edges),2))*edges[0][0], dtype=torch.long)
            data = Data(x=features, edge_index=edges_tensor.t().contiguous())
            dataset.append(data)
            no_graph = no_graph + 1
            node_idx = nodes[no_graph][:]
            node_set = set(node_idx)
            edges = []
            edges.append(graph[i][:])
    # features = torch.tensor(np.random.rand(len(node_idx), 1), dtype=torch.float)
    features = torch.tensor(np.random(shape=(len(node_idx), 1)), dtype=torch.float)
    edges_tensor = torch.tensor(edges-np.ones(shape=(len(edges),2))*edges[0][0], dtype=torch.long)
    data = Data(x=features, edge_index=edges_tensor.t().contiguous())
    dataset.append(data)
    # 각 그래프 node에 해당하는 엣지를 데이터로 넣어줌

    dataset_train = []
    for i in range(len(train)):
        dataset[train[i][0] - 1].y = torch.tensor([train[i][1]-1])
        dataset_train.append(dataset[train[i][0] - 1])

    dataset_test = []
    for i in range(len(test)):
        dataset_test.append(dataset[test[i][0] - 1])
    # Training set에는 라벨을 추가해주고, Test set도 제작


    torch.save(dataset_train, 'train.pt')
    torch.save(dataset_test, 'test.pt')

def load():
    dataset_test = torch.load('test.pt')
    dataset_train = torch.load('train.pt')


    return dataset_train, dataset_test

if __name__ == '__main__':
    generation()
