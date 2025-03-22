import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor

from data_stuff import ActionChunkDataset

acds = ActionChunkDataset()
acds.all_chunks = acds.all_chunks * 2 - 1 # move from [0, 1] tp [-1, 1]
print(acds.all_chunks)
action_data = acds.all_chunks.numpy()

print(f"Fitting tokenizer")
tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


tokens = tokenizer(action_data)

decoded_actions = tokenizer.decode(tokens, time_horizon=20, action_dim=7)

decoded_actions = torch.from_numpy(decoded_actions)

print(f"L2 loss was {F.mse_loss(action_data, decoded_actions)}")
