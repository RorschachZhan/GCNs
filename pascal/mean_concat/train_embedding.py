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

from model import Encoder, Model, drop_feature
from eval import label_classification
from data_utils import loadAllData
import numpy as np
import networkx as nx


def train(model: Model, x, edge_index, motif_dict, motifs_num):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1, motif_dict, motifs_num)
    z2 = model(x_2, edge_index_2, motif_dict, motifs_num)

    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model: Model, x, edge_index, y, motifs, motifs_num, final=False):
    model.eval()
    z = model(x, edge_index, motifs, motifs_num)
    ratio = 0.1
    if args.dataset == 'email':
        ratio = 0.5
    return label_classification(z, y, ratio=ratio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Polblogs')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='/home/dell3/PengY/GCL_MOTIF/config.yaml')
    parser.add_argument('--withone', type=int, default=0)#0表示不要1阶motif
    args = parser.parse_args()
    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)
    for datasets in ["polblogs"]:
        args.withone = 1
        allx, ally, graph, motifs_all, motifs_num = loadAllData(datasets, args.withone)
        lr = 0.001
        tau = 0.4
        config = yaml.load(open(args.config), Loader=SafeLoader)['Cora']
        config["learning_rate"] = lr
        config["tau"] = tau
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        learning_rate = lr
        num_hidden = config['num_hidden']
        num_proj_hidden = config['num_proj_hidden']
        activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
        base_model = ({'GCNConv': GCNConv})[config['base_model']]
        num_layers = config['num_layers']
        drop_edge_rate_1 = config['drop_edge_rate_1']
        drop_edge_rate_2 = config['drop_edge_rate_2']
        drop_feature_rate_1 = config['drop_feature_rate_1']
        drop_feature_rate_2 = config['drop_feature_rate_2']
        num_epochs = config['num_epochs']
        weight_decay = config['weight_decay']

        motifs_all = motifs_all.to(device)
        motifs_num = motifs_num.to(device)
        allylabel = []
        for item in ally:
            allylabel.append(np.argmax(item))

        graph = nx.from_dict_of_lists(graph)
        edges_index = torch.tensor(list(nx.edges(graph))).T

        
        allx = torch.tensor(allx.A, dtype=torch.float32).to(device)
        ally = torch.tensor(allylabel).to(device)
        edges_index = edges_index.to(device)
        for _ in range(1):#重复执行20次
            torch.manual_seed(config['seed'])
            random.seed(12345)
            encoder = Encoder(allx.shape[1], num_hidden, activation,
                            base_model=base_model, k=num_layers).to(device)
            model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            loss = 0
            best = 1e9
            best_t = 0
            with open("/home/dell3/PengY/GCL_MOTIF/mean_concat/" + datasets + ".txt", 'a+') as res_file:
                # res_file.write(str(config) + "\n")
                for epoch in range(1, 2001):
                    loss = train(model, allx, edges_index, motifs_all, motifs_num)
                    print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss))
                    if loss < best:
                        best = loss
                        best_t = epoch
                        cnt_wait = 0
                        torch.save(model.state_dict(), '/home/dell3/PengY/GCL_MOTIF/mean_concat/model.pkl')
                    else:
                        cnt_wait += 1

                    if cnt_wait == config['patience']:
                        print('Early stopping!')
                        res_file.write(f'Epoch={epoch}, Early stopping!\n')
                        break
                print('Loading {}th epoch'.format(best_t))
                res_file.write(f'Loading {best_t}th epoch\n')
                model.load_state_dict(torch.load('/home/dell3/PengY/GCL_MOTIF/mean_concat/model.pkl'))
                res = test(model, allx, edges_index, ally, motifs_all, motifs_num, final=True)
                res_file.write(f'Epoch={best_t} F1Mi={res["F1Mi"]} F1Ma={res["F1Ma"]}\n')

                embedding = model(allx, edges_index, motifs_all, motifs_num)
                test_embedding = embedding[-1000:]
                test_labels = ally[-1000:]
                torch.save(test_embedding, '/home/dell3/PengY/GCL_MOTIF/mean_concat/embedding.pt')
                torch.save(test_labels, "/home/dell3/PengY/GCL_MOTIF/mean_concat/label.pt")
                
                print(f'Epoch={best_t} F1Mi={res["F1Mi"]} F1Ma={res["F1Ma"]}')
            