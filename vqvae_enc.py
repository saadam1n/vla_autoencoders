import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_stuff import ActionChunkDataset
from testing_framework import TrajectoryNeuralEncoder
from transformers import AutoProcessor

import dct

from components import *

class VQEncoder(TrajectoryNeuralEncoder):
    """
    time_horizon - number of actions within action chunk
    action_dim - number of dims per action
    """
    def __init__(self, time_horizon, action_dim, num_embeddings, embedding_dim, num_tokens):
        super(VQEncoder, self).__init__()

        self.action_dim = action_dim
        self.time_horizon = time_horizon
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens
        self.latent_features = self.num_tokens * self.embedding_dim
        self.total_chunk_size = self.action_dim * self.time_horizon

        self.encoder = FeedForwardGELU(features_in=self.total_chunk_size, features_out=self.latent_features, expansion_ratio=8)

        self.vq = VQVAEBlock(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim)

        self.decoder = FeedForwardGELU(features_in=self.latent_features, features_out=self.total_chunk_size, expansion_ratio=8)

    def fit(self, acds, **kwargs):
        super().neural_train(acds)

        self.eval()

    """
    Don't return VQ loss for the regular encode function
    """
    def encode(self, data):
        enc, _ = self.differentiable_encode(data)

        return enc

    def decode(self, data):
        latent = data

        dec = self.differentiable_decode(latent)

        return dec

    def tokenization_prepare_encode(self, x : torch.Tensor):
        raise NotImplementedError()

    
    def tokenization_prepare_decode(self, data):
        raise NotImplementedError()

    def differential_encode_decode(self, x : torch.Tensor):
        # x is (N, L, C) format
        # where L = time_horizon and C = action_dim
        # we want (N, C, L) format (before we inevitably flatten it)

        xq, vqloss = self.differentiable_encode(x)

        xd = self.differentiable_decode(xq)

        return xd.view_as(x), vqloss
    
    def differentiable_encode(self, x : torch.Tensor) -> torch.Tensor:
        xf = x.flatten(1)

        xe = self.encoder(xf).unflatten(1, (self.num_tokens, self.embedding_dim))

        xq, vqloss = self.vq(xe)

        xq = xq.flatten(1)

        return xq, vqloss
    
    def differentiable_decode(self, xq : torch.Tensor) -> torch.Tensor:
        xd = self.decoder(xq)

        xd = xd.view(-1, self.time_horizon, self.action_dim)

        return xd