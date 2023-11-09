import torch
from torch_geometric.nn import GCNConv, SAGEConv, ChebConv, GATConv, GINConv
from torch_geometric.nn import SAGPooling, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F


class Siamese(torch.nn.Module):
    def __init__(self, args):
        super(Siamese, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio)

        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = SAGPooling(self.nhid, ratio=self.pooling_ratio)

        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool3 = SAGPooling(self.nhid, ratio=self.pooling_ratio)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.num_classes)
        self.lin2_1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2_2 = torch.nn.Linear(self.nhid, self.num_classes)
        self.lin3_1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin3_2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3_3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

    def forward_once(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch, None)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch, None)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch, None)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3
        x = x.view(x.size()[0], -1)
        x = F.relu(self.lin3_1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin3_2(x))  # 64
        x_norm = F.normalize(x, p=2, dim=1)
        x = self.lin3_3(x)
        return [x_norm, x]

    def forward(self, inputs1, inputs2):
        output1 = self.forward_once(inputs1)
        output2 = self.forward_once(inputs2)
        return output1, output2