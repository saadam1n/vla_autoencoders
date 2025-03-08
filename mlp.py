import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleLayerMLP(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio, norm_layer = nn.BatchNorm1d):
        super(DoubleLayerMLP, self).__init__()

        self.latent_channels = in_channels * expansion_ratio

        self.mlp = nn.Sequential(
            norm_layer(in_channels),
            nn.Linear(in_channels, self.latent_channels),
            nn.ReLU(),
            norm_layer(self.latent_channels),
            nn.Linear(self.latent_channels, out_channels)
        )

        self.skip = nn.Sequential(
            norm_layer(in_channels),
            nn.Linear(in_channels, out_channels)
        )

    def forward(self, x):
        return self.mlp(x) + self.skip(x)
    
    def transpose_output(self, indices):
        print(f"TPOS PREV {indices.shape}")
        self.mlp[4].weight.data = self.mlp[4].weight.data[indices, :]
        self.mlp[4].bias.data = self.mlp[4].bias.data[indices]
        self.skip[1].weight.data = self.skip[1].weight.data[indices, :]
        self.skip[1].bias.data = self.skip[1].bias.data[indices]

    def transpose_input(self, indices):
        self.mlp[1].weight.data = self.mlp[1].weight.data[:, indices]
        self.skip[1].weight.data = self.skip[1].weight.data[:, indices]

        self.mlp[0].running_mean.data = self.mlp[0].running_mean.data[indices] 
        self.mlp[0].running_var.data = self.mlp[0].running_var.data[indices] 
        self.mlp[0].weight.data = self.mlp[0].weight.data[indices]
        self.mlp[0].bias.data = self.mlp[0].bias.data[indices]

        self.skip[0].running_mean.data = self.skip[0].running_mean.data[indices] 
        self.skip[0].running_var.data = self.skip[0].running_var.data[indices] 
        self.skip[0].weight.data = self.skip[0].weight.data[indices]
        self.skip[0].bias.data = self.skip[0].bias.data[indices]