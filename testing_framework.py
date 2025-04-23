import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from data_stuff import ActionChunkDataset

class TrajectoryEncoder:
    def __init__(self):
        super().__init__()

    """
    It is assumed that fit takes care of *all* the training work 
    """
    def fit(self, acds : ActionChunkDataset, **kwargs):
        raise NotImplementedError("fit method not implemented!")
    
    def encode(self, data) -> torch.Tensor:
        raise NotImplementedError("encode method not implemented")
    
    def decode(self, data) -> torch.Tensor:
        raise NotImplementedError("decode method not implemented")
    
    def differential_encode_decode(self, data) -> torch.Tensor:
        raise NotImplementedError("differential_encode_decode not implemented! this encoder might not support such operation!")


class TrajectoryNeuralEncoder(TrajectoryEncoder, nn.Module):
    def __init__(self):
        super(TrajectoryNeuralEncoder, self).__init__()
        TrajectoryEncoder.__init__(self)
        nn.Module.__init__(self)

    def neural_train(self, acds : ActionChunkDataset):
        acds.all_chunks = acds.all_chunks.to("cuda")
        dataloader = torch.utils.data.DataLoader(
            acds,
            batch_size=512,
            shuffle=True
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.9)

        num_epochs = 256
        for i in range(num_epochs):

            print(f"Processing training for epoch {i}")

            sum_train_loss, sum_test_loss = 0, 0
            num_train_batches, num_test_batches = 0, 0

            self.train()
            acds.train_mode = True
            for j, sample in enumerate(dataloader):
                optimizer.zero_grad()

                sample = sample.to("cuda")

                output = self.differential_encode_decode(sample)

                loss = torch.nn.functional.l1_loss(output, sample)
                loss.backward()

                optimizer.step()

                sum_train_loss += loss.item()
                num_train_batches += 1

            print(f"Average L1 loss across all train samples was {sum_train_loss / num_train_batches}")

            print()

            print(f"Processing eval for epoch {i}")

            self.eval()
            acds.train_mode = False
            for j, sample in enumerate(dataloader):
                sample = sample.to("cuda")

                output = self.differential_encode_decode(sample)

                loss = F.mse_loss(output, sample)

                sum_test_loss += loss.item()
                num_test_batches += 1

            print(f"Average L2 loss across all eval samples was {sum_test_loss / num_test_batches}")

            print("\n\n\n")

            scheduler.step()