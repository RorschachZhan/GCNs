import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader
import time
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv
import scipy.sparse as sp

from model import Encoder, Model, drop_feature
from eval import label_classification
from data_utils import loadAllData
import numpy as np
import networkx as nx
from layer import GraphConvolution


def train(model: Model, x, motif_dict, motifs_num):
    model.train()
    optimizer.zero_grad()

    x_1 = drop_feature(x, 0.6)
    x_2 = drop_feature(x, 0.8)
    z1 = model(x_1, motif_dict, motifs_num)
    z2 = model(x_2, motif_dict, motifs_num)

    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model: Model, x, y, motifs, motifs_num, final=False):
    model.eval()
    z = model(x, motifs, motifs_num)
    ratio = 0.1
    if args.dataset == 'email':
        ratio = 0.5
    return label_classification(z, y, ratio=ratio)

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5AD^0.5
def sparse_to_tuple(sparse_mx):
    """
    Convert sparse matrix to tuple representation.
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Polblogs')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='/home/dell3/PengY/GCL_MOTIF/config.yaml')
    parser.add_argument('--withone', type=int, default=0)#0表示不要1阶motif
    args = parser.parse_args()
    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)
    for dataset in ["email"]:
        args.withone = 1
        lr = 0.001
        tau = 0.4
        config = yaml.load(open(args.config), Loader=SafeLoader)['Cora']
        config["learning_rate"] = lr
        config["tau"] = tau

        learning_rate = lr
        num_hidden = config['num_hidden']
        num_proj_hidden = config['num_proj_hidden']
        activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
        base_model = ({'GCNConv': GraphConvolution})[config['base_model']]
        num_layers = config['num_layers']

        drop_edge_rate_1 = config['drop_edge_rate_1']
        drop_edge_rate_2 = config['drop_edge_rate_2']
        drop_feature_rate_1 = config['drop_feature_rate_1']
        drop_feature_rate_2 = config['drop_feature_rate_2']
        num_epochs = config['num_epochs']
        weight_decay = config['weight_decay']

        allx, ally, graph, motifs_all, motifs_num = loadAllData(dataset, args.withone)

        allylabel = []
        for item in ally:
            allylabel.append(np.argmax(item))

        graph = nx.from_dict_of_lists(graph)

        edges_index = torch.tensor(list(nx.edges(graph))).T


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        allx = torch.tensor(allx.A, dtype=torch.float32).to(device)
        ally = torch.tensor(allylabel).to(device)
        motifs_all = motifs_all.to(device)
        motifs_num = motifs_num.to(device)
        num_features_nonzero = torch.nonzero(allx).shape[0]
        for _ in range(20):#重复执行20次
            torch.manual_seed(config['seed'])
            random.seed(12345)
            encoder = Encoder(allx.shape[1], num_hidden, activation,num_features_nonzero,
                            base_model=base_model, k=num_layers).to(device)
            model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            loss = 0
            best = 1e9
            best_t = 0
            with open("/home/dell3/PengY/GCL_MOTIF/mean_replace/" + dataset + ".txt", 'a+') as res_file:
                for epoch in range(1, 2001):
                    loss = train(model, allx, motifs_all, motifs_num)
                    print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss))
                    if loss < best:
                        best = loss
                        best_t = epoch
                        cnt_wait = 0
                        torch.save(model.state_dict(), '/home/dell3/PengY/GCL_MOTIF/mean_replace/model.pkl')
                    else:
                        cnt_wait += 1
                    if cnt_wait == config['patience']:
                        print('Early stopping!')
                        res_file.write(f'Epoch={epoch}, Early stopping!\n')
                        break
                print('Loading {}th epoch'.format(best_t))
                res_file.write(f'Loading {best_t}th epoch\n')
                model.load_state_dict(torch.load('/home/dell3/PengY/GCL_MOTIF/mean_replace/model.pkl'))
                res = test(model, allx, ally, motifs_all, motifs_num, final=True)
                res_file.write(f'Epoch={best_t} F1Mi={res["F1Mi"]} F1Ma={res["F1Ma"]}\n')
                print(f'Epoch={best_t} F1Mi={res["F1Mi"]} F1Ma={res["F1Ma"]}')
