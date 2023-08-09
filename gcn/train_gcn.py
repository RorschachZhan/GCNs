from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from model import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=64, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='acmv9', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=False, help='evaluation on test set.')
args = parser.parse_args()

if __name__ == "__main__":
    seed = 123
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # load data
    for rate in [""]:
    #for rate in ["0.1", "0.2", "0.3", "0.4", "0.5"]:

        for _ in range(1):

            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset, rate)
        # print('adj:', adj.shape)
        # print('features:', features.shape)
        # print('y:', y_train.shape, y_val.shape, y_test.shape)
        # print('mask:', train_mask.shape, val_mask.shape, test_mask.shape)

        # D^-1@X
        
            features = preprocess_features(features)  # [49216, 2], [49216], [2708, 1433]
            supports = preprocess_adj(adj)

            device = torch.device('cuda:0')
            train_label = torch.from_numpy(y_train).long().to(device)
            num_classes = train_label.shape[1]
            train_label = train_label.argmax(dim=1)
            train_mask = torch.from_numpy(train_mask.astype(np.int)).to(device)
            val_label = torch.from_numpy(y_val).long().to(device)
            val_label = val_label.argmax(dim=1)
            #val_mask = torch.from_numpy(val_mask.astype(np.int)).to(device)
            test_label = torch.from_numpy(y_test).long().to(device)
            test_label = test_label.argmax(dim=1)
            test_mask = torch.from_numpy(test_mask.astype(np.int)).to(device)

            i = torch.from_numpy(features[0]).long().to(device)
            v = torch.from_numpy(features[1]).to(device)
            feature = torch.sparse.FloatTensor(i.t(), v, features[2]).to(device)

            i = torch.from_numpy(supports[0]).long().to(device)  # 边
            v = torch.from_numpy(supports[1]).to(device)  # 权重
            support = torch.sparse.FloatTensor(i.t(), v, supports[2]).float().to(device)

            print('x :', feature)
            print('sp:', support)
            num_features_nonzero = feature._nnz()
            feat_dim = feature.shape[1]
            net = GCN(feat_dim, num_classes, num_features_nonzero)
            net.to(device)
            optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

            net.train()
            for epoch in range(args.epochs):

                out = net((feature.float(), support))
                out = out[0]
                loss = masked_loss(out, train_label, train_mask)
                loss += args.weight_decay * net.l2_loss()

                acc = masked_acc(out, train_label, train_mask)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch % 10 == 0:
                    # with open("/home/dell3/PengY/gcn/" + str(args.dataset) + ".txt", 'a+') as f:
                        print(epoch, loss.item(), acc.item())
                        # f.write(f'epoch={epoch}, loss={loss.item()}, acc={acc.item()}\n')

            net.eval()

            out = net((feature, support))
            out = out[0]
            acc = masked_acc(out, test_label, test_mask)
            print('test:', acc.item())
            #with open("/home/dell3/PengY/gcn/" + str(args.dataset) + rate + ".txt", 'a+') as f:
                #f.write(f'test: acc={acc.item()}\n')
