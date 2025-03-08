import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mlp import DoubleLayerMLP

chunk_size = 20

files = ["low_dim_v141(4).hdf5", "low_dim_v141(3).hdf5", "low_dim_v141(2).hdf5", "low_dim_v141(1).hdf5", "low_dim_v141.hdf5"]

real_robot_path = "low_dim_v141(4).hdf5"

num_demos = 0

all_demos = []

for real_robot_path in files:
    with h5py.File(real_robot_path, 'r') as f:
        for demo in f['data']:
            num_demos += 1

            demo_pos = f['data'][demo]['obs']['robot0_eef_pos'][:]
            demo_rot = f['data'][demo]['obs']['robot0_eef_quat'][:]

            demo_eef_actions = np.concatenate((demo_pos, demo_rot), axis=1)

            total_num_chunks = ((demo_eef_actions.shape[0] - 1) // chunk_size + 1)
            round_up_size = chunk_size * total_num_chunks - demo_eef_actions.shape[0]

            all_demos.append(demo_eef_actions)
            all_demos.append(np.zeros((round_up_size, demo_eef_actions.shape[1])))

eef_raw_actions = np.concatenate(all_demos, axis=0)

# FAST normalization
sorted_coeffs = np.sort(eef_raw_actions, axis=0)
lower = sorted_coeffs[int(sorted_coeffs.shape[0] * 0.01)]
upper = sorted_coeffs[int(sorted_coeffs.shape[0] * 0.99)]
quantile_range = upper - lower
eef_actions = (eef_raw_actions - lower) / quantile_range


print(f"Num demos is {num_demos}")

print(f"Found shape {eef_actions.shape}")
eef_actions = np.reshape(eef_actions, (eef_actions.shape[0] // chunk_size, chunk_size, -1))

    


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

        self.encoder = nn.Sequential(
            DoubleLayerMLP(self.flattened_input_dim, self.latent_vector_dim, expansion_ratio=8),
            FSQ(l)
        )

        self.decoder = nn.ModuleList([
            DoubleLayerMLP(i + 1, self.flattened_input_dim, expansion_ratio=8) for i in range(self.latent_vector_dim)
        ])

    def forward(self, x : torch.Tensor):
        compressed = self.encoder(x.flatten(1))
        
        reconstructed = torch.stack([
            self.decoder[i](compressed[:, :i + 1]).view_as(x) for i in range(self.latent_vector_dim)
        ], dim=1)

        return reconstructed
    



model = ActionAutoencoder(eef_actions.shape[1], 16, 16).to("cuda")
input = torch.tensor(eef_actions, dtype=torch.float32, device="cuda", requires_grad=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=128, T_mult=2, eta_min=0.001) if False else torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.9)

num_epochs = 3000
for i in range(num_epochs):
    optimizer.zero_grad()

    output = model(input)

    expanded_input = input.unsqueeze(1).expand(-1,  model.latent_vector_dim, -1, -1)

    loss = torch.nn.functional.mse_loss(output, expanded_input)
    loss.backward()

    optimizer.step()

    print(f"Fro loss at epoch {i} was {torch.nn.functional.mse_loss(output[:, -1, :, :], expanded_input[:, -1, :, :]).item()}")
