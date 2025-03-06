import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

class DoubleLayerMLP(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio):
        super(DoubleLayerMLP, self).__init__()

        self.latent_channels = in_channels * expansion_ratio

        self.batch_norm = nn.BatchNorm1d(in_channels)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, self.latent_channels),
            nn.ReLU(),
            nn.BatchNorm1d(self.latent_channels),
            nn.Linear(self.latent_channels, out_channels)
        )

        self.skip = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        xn = self.batch_norm(x)
        return self.mlp(xn) + self.skip(xn)
    
    def transpose_output(self, indices):
        print(f"TPOS PREV {indices.shape}")
        self.mlp[3].weight.data = self.mlp[3].weight.data[indices, :]
        self.mlp[3].bias.data = self.mlp[3].bias.data[indices]
        self.skip.weight.data = self.skip.weight.data[indices, :]
        self.skip.bias.data = self.skip.bias.data[indices]

    def transpose_input(self, indices):
        self.mlp[0].weight.data = self.mlp[0].weight.data[:, indices]
        self.skip.weight.data = self.skip.weight.data[:, indices]

        self.batch_norm.running_mean.data = self.batch_norm.running_mean.data[indices] 
        self.batch_norm.running_var.data = self.batch_norm.running_var.data[indices] 
        self.batch_norm.weight.data = self.batch_norm.weight.data[indices]
        self.batch_norm.bias.data = self.batch_norm.bias.data[indices]

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


model = ActionAutoencoder(eef_actions.shape[1], 16, 16).to("cuda")
input = torch.tensor(eef_actions, dtype=torch.float32, device="cuda", requires_grad=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=128, T_mult=2, eta_min=0.001) if False else torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.9)

num_epochs = 3000
for i in range(num_epochs):
    optimizer.zero_grad()

    output = model(input)

    loss = torch.nn.functional.mse_loss(output, input)
    loss.backward()

    optimizer.step()

    print(f"Fro loss at epoch {i} was {math.sqrt(loss.item())}")

model.eval()


def get_value_grad():
    compressed_coeffs = model.encode(input)
    compressed_coeffs.retain_grad()

    traj = model.decode(compressed_coeffs, input)

    fake_loss = torch.nn.functional.mse_loss(traj, torch.zeros_like(traj)).backward()

    return compressed_coeffs

value_grad = get_value_grad().grad.abs().sum(dim=0)
print(value_grad)

_, indices = torch.sort(value_grad, descending=True)
indices = indices.flatten()

with torch.no_grad():
    model.encoder.transpose_output(indices)
    model.decoder.transpose_input(indices)

print(f"Take two: {get_value_grad().grad.abs().sum(dim=0)}")




with torch.no_grad():
    compressed_coeffs = model.encode(input)

    traj = model.decode(compressed_coeffs, input)



    error = (traj - input).detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    show_coeffs = compressed_coeffs.detach().cpu().numpy()
    ax.scatter(xs=show_coeffs[:, 0], ys=show_coeffs[:, 1], zs=show_coeffs[:, 2], s=0.1, marker='o')
    plt.show()

    ax.scatter(xs=error[:, 0], ys=error[:, 1], zs=error[:, 2], s=0.1, marker='o')
    plt.show()
