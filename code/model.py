import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F

class scABiGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(scABiGNN, self).__init__()

        emb_channels = 512
        self.emb = GCNConv(in_channels, emb_channels)

        self.att1 = GATConv(emb_channels, emb_channels, heads=1)
        self.norm1 = nn.LayerNorm(emb_channels)
        hidden_channel1 = 256
        self.linear1 = torch.nn.Sequential(
            torch.nn.Linear(emb_channels, hidden_channel1),
            torch.nn.ReLU(),
        )

        self.att2 = GATConv(hidden_channel1, hidden_channel1, heads=1)
        self.norm2 = nn.LayerNorm(hidden_channel1)
        hidden_channel2 = 128
        self.linear2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channel1, hidden_channel2),
            torch.nn.ReLU(),
        )

        conv_channels = 64
        self.conv = GCNConv(hidden_channel2, conv_channels)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(conv_channels, conv_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(conv_channels // 2, out_channels)
        )

    def forward(self, x, edge_index):
        x = self.emb(x, edge_index)

        att1 = self.att1(x, edge_index)
        x = self.norm1(x + att1)
        x = self.linear1(x)


        att2 = self.att2(x, edge_index)
        x = self.norm2(x + att2)
        x = self.linear2(x)

        x = self.conv(x, edge_index)
        x = self.classifier(x)

        log_probs = F.log_softmax(x, dim=1)
        return log_probs