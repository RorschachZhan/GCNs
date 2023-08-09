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
from torch import Tensor
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.nn import GCNConv

from model import Encoder, Model, drop_feature
from eval import label_classification
from data_utils import loadAllData
from model import Model
import numpy as np
import networkx as nx
from torch_sparse import coalesce


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1
    else:
        return max(edge_index.size(0), edge_index.size(1))


def filter_adj(row, col, edge_attr, mask):
    return row[mask], col[mask], None if edge_attr is None else edge_attr[mask]


def dropout_adj(edge_index, edge_attr=None, p=0.5, lower=0.1, upper=1, force_undirected=False,
                num_nodes=None, training=True, is_adaptive=False):
    if p < 0. or p > 1.:
        raise ValueError('Dropout probability has to be between 0 and 1, '
                         'but got {}'.format(p))

    if not training or p == 0.0:
        return edge_index, edge_attr

    N = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    if force_undirected:
        row, col, edge_attr = filter_adj(row, col, edge_attr, row < col)

    if is_adaptive:  # 根据边权重，计算drop概率
        probability = torch.where(torch.gt(edge_attr, 50), torch.zeros_like(edge_attr).fill_(50), edge_attr)
        a = lower
        b = upper
        k = (b - a) / (torch.max(edge_attr) - torch.min(edge_attr))
        probability = a + k * (edge_attr - torch.min(edge_attr))
        mask = torch.bernoulli(probability).to(torch.bool)
    else:
        mask = edge_index.new_full((row.size(0),), 1 - p, dtype=torch.float)
        mask = torch.bernoulli(mask).to(torch.bool)

    row, col, edge_attr = filter_adj(row, col, edge_attr, mask)

    if force_undirected:
        edge_index = torch.stack(
            [torch.cat([row, col], dim=0),
             torch.cat([col, row], dim=0)], dim=0)
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
    else:
        edge_index = torch.stack([row, col], dim=0)

    return edge_index, edge_attr


def train(model: Model, bxent, lbl, x, edge_index, edge_weight, is_edge_weight, is_adaptive, lower=0.1, upper=1):
    model.train()
    optimizer.zero_grad()
    edge_index_1, edge_weight_1 = dropout_adj(edge_index, edge_weight, drop_edge_rate_1, lower, upper,
                                              is_adaptive=is_adaptive)
    edge_index_2, edge_weight_2 = dropout_adj(edge_index, edge_weight, drop_edge_rate_2, lower, upper,
                                              is_adaptive=is_adaptive)
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    # x_1 = x
    # x_2 = x

    if is_edge_weight:
        logits, __, __ = model(x_1, edge_index_1, edge_weight_1, x_2, edge_index_2, edge_weight_2, None)
    else:
        logits, __, __ = model(x_1, edge_index_1, None, x_2, edge_index_2, None, None)
    loss = b_xent(logits, lbl.reshape(1, -1))
    loss.backward()
    optimizer.step()

    return loss.item()


def train_test(model, x, edge_index, edge_weight, y, is_edge_weight, final=False):
    model.eval()
    if is_edge_weight:
        z, c = model.embed(x, edge_index, edge_weight)
    else:
        z, c = model.embed(x, edge_index, None)
    ratio = 0.1
    if args.dataset == 'email_motif':
        ratio = 0.6
    return label_classification(z, y, ratio=ratio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Polblogs')
    parser.add_argument('--gpu_id', '-g', type=int, default=0)
    parser.add_argument('--config', type=str, default='./config.yaml')
    parser.add_argument('--is_edge_weight', '-w', action='store_false')
    parser.add_argument('--is_adaptive', '-a', action='store_false')
    args = parser.parse_args()
    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)
    is_edge_weight = args.is_edge_weight
    is_adaptive = args.is_adaptive
    # for datasets in [["acmv9", "Football"],["citationv1", "Football"], ["dblpv7", "Football"],["amazon-photo", "Football"],["amazon-computers", "Football"]]:
    for datasets in [["amazon-computers", "Football"]]:
        config = yaml.load(open(args.config), Loader=SafeLoader)[datasets[1]]

        torch.manual_seed(config['seed'])
        random.seed(config['seed'])

        learning_rate = config['learning_rate']
        num_hidden = config['num_hidden']
        num_proj_hidden = config['num_proj_hidden']
        activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
        base_model1 = ({'GCNConv': GCNConv})[config['base_model']]
        base_model2 = ({'GCNConv': GCNConv})[config['base_model']]
        num_layers = config['num_layers']

        drop_edge_rate_1 = config['drop_edge_rate_1']
        drop_edge_rate_2 = config['drop_edge_rate_2']
        drop_feature_rate_1 = config['drop_feature_rate_1']
        drop_feature_rate_2 = config['drop_feature_rate_2']
        tau = config['tau']
        weight_decay = config['weight_decay']
        lower = config['lower']
        upper = config['upper']
        str1 = "_weight" if is_edge_weight else ""
        str2 = "_adaptive" if is_adaptive else ""
        res_path = "./results/" + datasets[0] + "_grace_" + str(learning_rate) + str1 + str2 + ".txt"
        # for lower in [0.2, 0.3, 0.4, 0.5]:
        # config['lower'] = lower
        with open(res_path, 'a+') as res_file:
            res_file.write(str(config) + "\n")
        for _ in range(20):
            allx, ally, edges, edges_weight = loadAllData(datasets[0])

            allylabel = []
            for item in ally:
                allylabel.append(np.argmax(item))
            edges_index = torch.tensor(edges).T
            edges_weight = torch.tensor(edges_weight).T
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # device = torch.device('cpu')
            allx = torch.tensor(allx.A, dtype=torch.float32).to(device)
            ally = torch.tensor(allylabel).to(device)
            edges_index = edges_index.to(device)
            edges_weight = edges_weight.to(device)
            lbl_1 = torch.ones(allx.shape[0] * 2)
            lbl_2 = torch.zeros(allx.shape[0] * 2)
            lbl = torch.cat((lbl_1, lbl_2), 0).reshape(1, -1)
            lbl = lbl.to(device)

            encoder1 = Encoder(allx.shape[1], num_hidden, activation,
                               base_model=base_model1, k=num_layers).to(device)
            encoder2 = Encoder(allx.shape[1], num_hidden, activation,
                               base_model=base_model2, k=num_layers).to(device)
            model = Model(encoder1, encoder2, num_hidden, num_proj_hidden, tau).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            b_xent = nn.BCEWithLogitsLoss()
            loss = 0
            best = 1e9
            best_t = 0
            with open(res_path, 'a+') as res_file:
                for epoch in range(1, 1001):
                    loss = train(model, b_xent, lbl, allx, edges_index, edges_weight, is_edge_weight, is_adaptive, lower, upper)
                    # if epoch % 10 == 0:
                    print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss))
                    if loss < best:
                        best = loss
                        best_t = epoch
                        cnt_wait = 0
                        #torch.save(model.state_dict(), './model' + str1 + str2 + '.pkl')
                        #torch.save(model, './model' + str1 + str2 + '.pkl')
                        torch.save(model,'./model_' + datasets[0]+'.pkl')
                    else:
                        cnt_wait += 1
                    if cnt_wait == config['patience']:
                        print('Early stopping!')
                        res_file.write(f'Epoch={epoch}, Early stopping!\n')
                        break
                #model=torch.load('./model' + str1 + str2 + '.pkl')
                model=torch.load('./model_' + datasets[0]+'.pkl')
                res = train_test(model, allx, edges_index, edges_weight, ally, is_edge_weight, final=True)
                #res_file.write(f'Epoch={best_t} F1Mi={res["F1Mi"]} F1Ma={res["F1Ma"]}\n')
                print(f'Epoch={best_t} F1Mi={res["F1Mi"]} F1Ma={res["F1Ma"]}')
