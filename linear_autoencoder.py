import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_stuff import ActionChunkDataset
from testing_framework import TrajectoryNeuralEncoder
from transformers import AutoProcessor

import dct

from components import *


class LinearAutoencoder(TrajectoryNeuralEncoder):
    """
    time_horizon - number of actions within action chunk
    action_dim - number of dims per action
    """
    def __init__(self, time_horizon, action_dim, vocab_size, num_tokens, roundlu_vocab = None):
        super(LinearAutoencoder, self).__init__()

        self.action_dim = action_dim
        self.time_horizon = time_horizon
        self.vocab_size = vocab_size
        self.num_tokens = num_tokens
        self.total_chunk_size = self.action_dim * self.time_horizon
        self.roundlu_vocab = None

        self.encoder = FeedForwardReLU(features_in=self.total_chunk_size, features_out=self.num_tokens, expansion_ratio=8, roundlu_vocab=self.roundlu_vocab)

        self.fsq = nn.Sequential(
            nn.BatchNorm1d(self.num_tokens), # switch running_mean and var
            FiniteScalarQuantization(self.vocab_size),
        )

        self.decoder = FeedForwardReLU(features_in=self.num_tokens, features_out=self.total_chunk_size, expansion_ratio=8)

        self.tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

    def fit(self, acds, **kwargs):
        super().neural_train(acds)

        self.eval()

        # neural transposition
        all_input = acds.all_chunks.to("cuda")

        # neuron transposition
        expected_out = self.differential_encode_decode(all_input)
        dropout_check = self.differential_encode_decode(all_input)

        print(f"Dropout check was {F.mse_loss(dropout_check, expected_out)}")

        expected_fsq = self.differentiable_encode(all_input)

        def get_value_grad():
            compressed_coeffs = self.differentiable_encode(all_input)
            compressed_coeffs.retain_grad()

            traj = self.differentiable_decode(compressed_coeffs).view_as(all_input)

            fake_loss = torch.nn.functional.mse_loss(traj, torch.zeros_like(traj)).backward()

            return compressed_coeffs

        value_grad = get_value_grad().grad.abs().sum(dim=0)
        print(value_grad)

        _, indices = torch.sort(value_grad, descending=True)
        indices = indices.flatten()

        with torch.no_grad():
            self.transpose(indices)

        actual_out = self.differential_encode_decode(all_input)
        actual_fsq = self.differentiable_encode(all_input)

        print(f"FSQ MSE loss from transpositon was {F.mse_loss(actual_fsq, expected_fsq[:, indices])}")
        print(f"TOT MSE loss from transpositon was {F.mse_loss(actual_out, expected_out)}")
        print(actual_fsq[0])
        print(expected_fsq[0, indices])
        print(expected_fsq[0])

        xd = self.differentiable_encode(all_input)
        xq_np = self.tokenization_prepare_encode(xd)

        self.tokenizer = self.tokenizer.fit(xq_np)

        xq_tokenized = self.tokenizer(xq_np)

        avg_seq_len = sum([len(tokens) for tokens in xq_tokenized]) / len(xq_tokenized)

        xd = self.differentiable_decode(
            self.tokenization_prepare_decode(xq_tokenized)
        ).view_as(all_input)



        bpe_avg_l2 = F.mse_loss(xd, all_input.to("cuda"))

        print(f"BPE len was {avg_seq_len} and loss was {bpe_avg_l2}")

    def encode(self, data):
        enc = self.differentiable_encode(data)

        enc = self.tokenizer(self.tokenization_prepare_encode(enc))

        avg_len = sum([len(tokens) for tokens in enc]) / len(enc)
        print(f"Linear Autoencoder: average tokenized length was {avg_len}")

        return enc

    def decode(self, data):
        latent = self.tokenization_prepare_decode(data)

        dec = self.differentiable_decode(latent)

        return dec

    def tokenization_prepare_encode(self, x : torch.Tensor):
        # tokenize entire dataset
        # move tokens [-1, 1]
        xq = (x * 2.0 / self.vocab_size).clamp(min=-1, max=1)

        # (N, C) -> (N, C, 1) to put everything in one DCT channel
        xq_np = xq.unsqueeze(1).detach().cpu().numpy()

        return xq_np
    
    def tokenization_prepare_decode(self, data):
        xd = torch.from_numpy(self.tokenizer.decode(data)).to("cuda").float() * self.vocab_size / 2.0
        
        xd = xd.squeeze(1)

        return xd

    def differential_encode_decode(self, x : torch.Tensor):
        # x is (N, L, C) format
        # where L = time_horizon and C = action_dim
        # we want (N, C, L) format (before we inevitably flatten it)

        xq = self.differentiable_encode(x)

        xd = self.differentiable_decode(xq)

        return xd.view_as(x)
    
    def differentiable_encode(self, x : torch.Tensor) -> torch.Tensor:
        xf = x.flatten(1)

        xe = self.encoder(xf)

        xq = self.fsq(xe)

        return xq
    
    def differentiable_decode(self, xq : torch.Tensor) -> torch.Tensor:
        xd = self.decoder(xq)

        xd = xd.view(-1, self.time_horizon, self.action_dim)

        return xd
    
    def transpose(self, indices : torch.Tensor):
        print(f"TRANSPOSITON SHAPE {indices.shape}")
        print(f"\tVAL{indices}")

        self.encoder.feed_forward[3].weight.data = self.encoder.feed_forward[3].weight.data[indices, :]
        self.encoder.feed_forward[3].bias.data = self.encoder.feed_forward[3].bias.data[indices]

        self.encoder.skip_linear[1].weight.data = self.encoder.skip_linear[1].weight.data[indices, :]
        self.encoder.skip_linear[1].bias.data = self.encoder.skip_linear[1].bias.data[indices]

        self.fsq[0].running_mean.data = self.fsq[0].running_mean.data[indices]
        self.fsq[0].running_var.data = self.fsq[0].running_var.data[indices]
        self.fsq[0].weight.data = self.fsq[0].weight.data[indices]
        self.fsq[0].bias.data = self.fsq[0].bias.data[indices]
    
        self.decoder.feed_forward[0].running_mean.data = self.decoder.feed_forward[0].running_mean.data[indices]
        self.decoder.feed_forward[0].running_var.data = self.decoder.feed_forward[0].running_var.data[indices]
        self.decoder.feed_forward[0].weight.data = self.decoder.feed_forward[0].weight.data[indices]
        self.decoder.feed_forward[0].bias.data = self.decoder.feed_forward[0].bias.data[indices]
        self.decoder.feed_forward[1].weight.data = self.decoder.feed_forward[1].weight.data[:, indices]

        self.decoder.skip_linear[0].running_mean.data = self.decoder.skip_linear[0].running_mean.data[indices]
        self.decoder.skip_linear[0].running_var.data = self.decoder.skip_linear[0].running_var.data[indices]
        self.decoder.skip_linear[0].weight.data = self.decoder.skip_linear[0].weight.data[indices]
        self.decoder.skip_linear[0].bias.data = self.decoder.skip_linear[0].bias.data[indices]
        self.decoder.skip_linear[1].weight.data = self.decoder.skip_linear[1].weight.data[:, indices]

