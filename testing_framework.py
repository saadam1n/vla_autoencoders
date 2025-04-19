import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from data_stuff import ActionChunkDataset

class TrajectoryEncoder:
    def __init__(self):
        pass

    """
    It is assumed that fit takes care of *all* the training work 
    """
    def fit(self, acds : ActionChunkDataset, **kwargs):
        raise NotImplementedError("fit method not implemented!")
    
    def encode(self, data) -> torch.Tensor:
        raise NotImplementedError("encode method not implemented")
    
    def decode(self, data) -> torch.Tensor:
        raise NotImplementedError("decode method not implemented")
    

def train(model : TrajectoryEncoder, acds : ActionChunkDataset):
    dataloader = torch.utils.data.DataLoader(
        acds,
        batch_size=512,
        shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.9)

    num_epochs = 128
    for i in range(num_epochs):

        print(f"Processing training for epoch {i}")

        sum_train_loss, sum_test_loss = 0.0
        num_train_batches, num_test_batches = 0

        model.train()
        acds.train_mode = True
        for j, sample in enumerate(dataloader):
            optimizer.zero_grad()

            sample = sample.to("cuda")

            enc = model.encode(sample)
            output = model.decode(enc)

            loss = torch.nn.functional.l1_loss(output, sample)
            loss.backward()

            optimizer.step()

            sum_train_loss += loss.item()
            num_train_batches += 1

        print(f"Average L1 loss across all train samples was {sum_train_loss / num_train_batches}")

        print()

        print(f"Processing eval for epoch {i}")

        model.eval()
        acds.train_mode = False
        for j, sample in enumerate(dataloader):
            sample = sample.to("cuda")

            enc = model.encode(sample)
            output = model.decode(enc)

            loss = F.mse_loss(output, sample)

            sum_test_loss += loss.item()
            num_test_batches += 1

        print(f"Average L2 loss across all eval samples was {sum_test_loss / num_test_batches}")

        print("\n\n\n")

        scheduler.step()