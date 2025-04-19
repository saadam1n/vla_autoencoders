# this file fits a model and then evaluates it on a training data set
import testing_framework

from hash_encoder import DoubleHashEncoder
from data_stuff import ActionChunkDataset

import torch.nn.functional as F

acds = ActionChunkDataset()

dhe = DoubleHashEncoder(num_entries=4096, quant_bins=16)
dhe.fit(acds)

test_data = acds.test_split()

latent = dhe.encode(test_data)
reconstructed = dhe.decode(latent)

print(f"MSE loss was {F.mse_loss(reconstructed, test_data)}")
