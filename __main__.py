# this file fits a model and then evaluates it on a training data set
import testing_framework

from hash_encoder import DoubleHashEncoder
from baseline import FastEncoder
from linear_autoencoder import LinearAutoencoder
from data_stuff import ActionChunkDataset
from hashed_linear_nt import HashedLinearAutoencoder
from vqvae_enc import VQEncoder

import torch
import torch.nn.functional as F

acds = ActionChunkDataset()

#model = DoubleHashEncoder(num_entries=4096, quant_bins=16)
#model = LinearAutoencoder(time_horizon=20, action_dim=7, vocab_size=1024, num_tokens=16)
model = VQEncoder(time_horizon=20, action_dim=7, num_embeddings=1024, embedding_dim=64, num_tokens=16)
model = model.to("cuda")
model.fit(acds)

test_data = acds.test_split()

with torch.no_grad():
    latent = model.encode(test_data)
    reconstructed = model.decode(latent)

    print(f"MSE loss was {F.mse_loss(reconstructed, test_data)}")
