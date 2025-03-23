import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch.utils
import torch.utils.data
from mlp import DoubleLayerMLP
from data_stuff import ActionChunkDataset


chunk_size = 20

class FSQ(nn.Module):
    def __init__(self, l):
        super(FSQ, self).__init__()

        self.quantization_granularity = l

    def forward(self, x: torch.Tensor):
        z = x.tanh() * self.quantization_granularity / 2.0

        zhat = z + (z.round() - z).detach()

        return zhat

"""
Input: N x 7 matrix where N is action size
Output: 1 x M vector (M << n x 7)
"""
class ActionAutoencoder(nn.Module):
    def __init__(self, n, m, l):
        super(ActionAutoencoder, self).__init__()

        self.flattened_input_dim = n * 7
        self.expansion_ratio = 2
        self.latent_vector_dim = m
        self.proj_dim = 512

        self.encoder = DoubleLayerMLP(self.flattened_input_dim, self.latent_vector_dim, expansion_ratio=8)

        self.fsq = FSQ(l)

        self.decoder = DoubleLayerMLP(self.latent_vector_dim, self.flattened_input_dim, expansion_ratio=8)


    def forward(self, x : torch.Tensor):
        coeffs = self.encode(x)
        traj = self.decode(coeffs, x)

        return traj
    
    def encode(self, x : torch.Tensor):
        xf = x.flatten(1)
        xe = self.encoder(xf)
        xq = self.fsq(xe)

        return xq

    def decode(self, xq : torch.Tensor, vl):
        xd = self.decoder(xq)
        xv = xd.view_as(vl)

        return xv



model = ActionAutoencoder(20, 12, 1024).to("cuda")

acds = ActionChunkDataset()
dataloader = torch.utils.data.DataLoader(
    acds,
    batch_size=256,
    shuffle=True
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)

num_epochs = 128
for i in range(num_epochs):

    print(f"Processing training for epoch {i}")

    model.train()
    acds.train_mode = True
    for j, sample in enumerate(dataloader):
        optimizer.zero_grad()

        sample = sample.to("cuda")

        output = model(sample)

        loss = torch.nn.functional.l1_loss(output, sample)
        loss.backward()

        optimizer.step()

        #print(f"\tL1 loss in btach {j} was {loss.item()}")

    print()

    print(f"Processing eval for epoch {i}")

    total_loss = 0.0
    num_eval_samples = 0

    model.eval()
    acds.train_mode = False
    for j, sample in enumerate(dataloader):
        optimizer.zero_grad()

        sample = sample.to("cuda")
        output = model(sample)

        loss = torch.nn.functional.mse_loss(output, sample)
        loss.backward()

        optimizer.step()

        #print(f"\tL2 loss in batch {j} was {loss.item()}")
        total_loss += loss.item() * output.shape[0]
        num_eval_samples += output.shape[0]

    print(f"Average L2 loss across all eval samples was {total_loss / num_eval_samples}")

    print("\n\n\n")

print("OUTPUT FOR LINEAR NT!")
