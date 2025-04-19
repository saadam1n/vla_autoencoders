import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor

from data_stuff import ActionChunkDataset

acds = ActionChunkDataset()
print(acds.all_chunks)
train_data = acds.train_split().numpy()
test_data = acds.test_split()

print(f"Fitting tokenizer")
tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
tokenizer = tokenizer.fit(train_data)

tokens = tokenizer(test_data)



if False:
    num_attempts = 0
    successful_attempts = 0
    for token in tokens:
        idk = [token]

        decoded = tokenizer.decode(idk, time_horizon=20, action_dim=7)

        if not np.all(decoded == 0):
            successful_attempts += 1

        num_attempts += 1

        print(f"{successful_attempts}/{num_attempts}")

decoded_actions = tokenizer.decode(tokens, time_horizon=20, action_dim=7)

decoded_actions = torch.from_numpy(decoded_actions)

[ print(f"L2 loss was {F.mse_loss(test_data[:, :, i], decoded_actions[:, :, i])}") for i in range(7) ]
print(f"L2 loss was {F.mse_loss(test_data, decoded_actions)}")

tot = 0
for token in tokens:
    tot += len(token)

tot /= len(tokens)
print(f"Avg tokens was {tot}")