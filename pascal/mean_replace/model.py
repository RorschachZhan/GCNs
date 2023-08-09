import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import GraphConvolution


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, num_features_nonzero,
                 base_model=GraphConvolution, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels, num_features_nonzero)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels, num_features_nonzero))
        self.conv.append(base_model(2 * out_channels, out_channels, num_features_nonzero))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, motifs_all, motifs_num):
        for i in range(self.k):
            temp = [(torch.mm(motifs_all[mat], x).T / motifs_num[mat]).T for mat in range(len(motifs_all))]
            temp2 = torch.stack(temp, 0)
            x_agg = torch.mean(temp2, 0)
            x = self.activation(self.conv[i](x_agg))
        return x


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor, motifs_all, motifs_num) -> torch.Tensor:
        # temp = torch.matmul(motifs_all, x)
        # temp2 = temp/motifs_num

        # x_agg = torch.mean(torch.tensor([(torch.mm(motifs_all[mat], x)/motifs_num[mat]).numpy() for mat in range(len(motifs_all))]),0)
        # x_agg = torch.zeros(x.shape[0], x.shape[1])
        # for i in range(x.shape[0]):
        #     x_agg[i] = torch.mean(torch.tensor([torch.mean(x[motifs], 0).numpy() for motifs in motifs_all[i]]), 0)
        return self.encoder(x, motifs_all, motifs_num)
        # for i in range(len(motif_dict)):
        #     motifs_type = motif_dict[i]#[{},{},{},{},{}]5中类型motif
        #     motifs_emb = []
        #     for j in range(len(motifs_type)):
        #         motifs = motifs_type[j]#某种类型motif
        #         if len(motifs) == 0:
        #             continue
        #         motifs_type_emb = torch.zeros(x.shape[1])
        #         for motif in motifs:
        #             motifs_type_emb += sum(x[list(motif)])
        #         motifs_type_mean = motifs_type_emb/len(motifs)
        #         motifs_emb.append(motifs_type_mean)
        #     x_agg[i] = torch.mean(torch.tensor([t.numpy() for t in motifs_emb]).float(), 0)
        # return self.encoder(x_agg, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x
