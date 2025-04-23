# this file fits a model and then evaluates it on a training data set
import testing_framework

from hash_encoder import DoubleHashEncoder
from baseline import FastEncoder
from linear_autoencoder import LinearAutoencoder
from data_stuff import ActionChunkDataset
from hashed_linear_nt import HashedLinearAutoencoder

import torch.nn.functional as F

acds = ActionChunkDataset()

model = DoubleHashEncoder(num_entries=4096, quant_bins=16)
model = HashedLinearAutoencoder(time_horizon=20, action_dim=7, vocab_size=16, num_tokens=16)
model = model.to("cuda")
model.fit(acds)

test_data = acds.test_split()

latent = model.encode(test_data)
reconstructed = model.decode(latent)

print(f"MSE loss was {F.mse_loss(reconstructed, test_data)}")
