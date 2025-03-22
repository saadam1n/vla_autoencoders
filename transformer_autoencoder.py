import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import sys
from mlp import DoubleLayerMLP
from data_stuff import ActionChunkDataset

chunk_size = 20

class MultiHeadCasualAttention(nn.Module):
    def __init__(self, input_embedding_dim, num_heads, head_dim):
        super(MultiHeadCasualAttention, self).__init__()

        self.smax_mult = math.sqrt(input_embedding_dim)
        self.all_head_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.q_proj = nn.Sequential(
            nn.LayerNorm(input_embedding_dim),
            nn.Linear(input_embedding_dim, self.all_head_dim)
        )

        self.k_proj = nn.Sequential(
            nn.LayerNorm(input_embedding_dim),
            nn.Linear(input_embedding_dim, self.all_head_dim)
        )

        self.v_proj = nn.Sequential(
            nn.LayerNorm(input_embedding_dim),
            nn.Linear(input_embedding_dim, self.all_head_dim)
        )

        self.head_aggregator = nn.Linear(self.all_head_dim, input_embedding_dim)

        self.mlp = DoubleLayerMLP(input_embedding_dim, input_embedding_dim, expansion_ratio=2, norm_layer=nn.LayerNorm)

    def forward(self, input_tokens):
        attended = self.mha(input_tokens)

        return self.mlp(attended)
    
    def mha(self, input_tokens):
        seq_len = input_tokens.shape[1]
        mask_offset = self.v_cache.shape[2] if self.v_cache is not None else 0

        # proj to get our Q, K, V
        q = self.q_proj(input_tokens).view(-1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(input_tokens).view(-1, seq_len, self.num_heads, self.head_dim).transpose(1, 2).transpose(2, 3)
        v = self.v_proj(input_tokens).view(-1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        self.k_cache = torch.cat((self.k_cache, k), dim=3) if self.k_cache is not None else k
        self.v_cache = torch.cat((self.v_cache, v), dim=2) if self.v_cache is not None else v

        # run attention on all of these matrices
        qkT = torch.matmul(q, self.k_cache) / self.smax_mult
        mask = torch.triu(torch.ones_like(qkT) * -999.0, diagonal=mask_offset + seq_len - 1)
        softmax_scores = F.softmax(qkT + mask, dim=3)
        attended_tokens = torch.matmul(softmax_scores, self.v_cache).transpose(1, 2).flatten(2)

        return self.head_aggregator(attended_tokens) + input_tokens

    def clear_kv(self):
        self.k_cache = None
        self.v_cache = None


class TransformerAutoencoderMHA(nn.Module):
    """
    Based on a GPT3-like decoderless architecture
    """
    def __init__(self, num_actions, token_dimension, num_embeddings, num_input_tokens, num_output_tokens):
        super(TransformerAutoencoderMHA, self).__init__()

        self.flattened_input_channels = 7
        self.flattened_input_dim = num_actions * self.flattened_input_channels
        self.conv_num_channels = 16
        self.conv_input_dim = num_actions * self.conv_num_channels
        self.token_dimension = token_dimension
        self.num_embeddings = num_embeddings
        self.num_output_tokens = num_output_tokens
        self.num_input_tokens = num_input_tokens

        self.conv = nn.Sequential(
            nn.BatchNorm1d(self.flattened_input_channels),
            nn.Conv1d(self.flattened_input_channels, self.conv_num_channels, kernel_size=5, padding=2)
        )

        self.encoder_dim = self.num_input_tokens * self.token_dimension
        self.tokenizer = nn.Sequential(
            DoubleLayerMLP(self.conv_input_dim, self.encoder_dim, expansion_ratio=2)
        )

        self.attention = nn.Sequential(
            MultiHeadCasualAttention(self.token_dimension, num_heads=8, head_dim=6),
            MultiHeadCasualAttention(self.token_dimension, num_heads=8, head_dim=6),
            MultiHeadCasualAttention(self.token_dimension, num_heads=8, head_dim=6),
            MultiHeadCasualAttention(self.token_dimension, num_heads=8, head_dim=6),
        )

        self.embedding_matrix = nn.Parameter(torch.randn(self.token_dimension, self.num_embeddings))

        self.positional_encoding = nn.Parameter(torch.randn(self.num_output_tokens, self.token_dimension))

        self.decoder = DoubleLayerMLP(self.num_output_tokens * self.token_dimension, self.flattened_input_dim, expansion_ratio=2)


        
    def forward(self, x : torch.Tensor):
        xc = self.conv(x.transpose(1, 2))

        xf = xc.flatten(1)
        input_tokens = self.tokenizer(xf).view(-1, self.num_input_tokens, self.token_dimension)

        for layer in self.attention:
            layer.clear_kv()
        self.attention(input_tokens)

        output_tokens = []
        for i in range(self.num_output_tokens):
            raw_next_token = self.positional_encoding[i].expand(x.shape[0], -1).unsqueeze(1)

            if i != 0 and False:
                raw_next_token = raw_next_token + output_tokens[i - 1]

            attended_next_token = self.attention(raw_next_token)

            embedding_probabilities = F.softmax(torch.matmul(attended_next_token, self.embedding_matrix).view(-1, self.num_embeddings), dim=1)
            selected_index = torch.searchsorted(embedding_probabilities.cumsum(dim=1), torch.rand_like(embedding_probabilities[:, 0]).unsqueeze(1)).clamp(min=0, max=self.num_embeddings - 1)

            selected_token = self.embedding_matrix[:, selected_index].permute((1, 2, 0))

            # gradient passing trick
            selected_token = selected_token + attended_next_token - attended_next_token.detach()

            output_tokens.append(selected_token)

        output_tokens = torch.cat(output_tokens, dim=1).flatten(1)
        traj = self.decoder(output_tokens).view_as(x)

        return traj

"""
Input: N x 7 matrix where N is action size
Output: 1 x M vector (M << n x 7)
"""
class TransformerAutoencoder(nn.Module):
    """
    Based on a GPT3-like decoderless architecture
    """
    def __init__(self, num_actions, token_dimension, num_embeddings, num_input_tokens, num_output_tokens):
        super(TransformerAutoencoder, self).__init__()

        self.flattened_input_channels = 6
        self.flattened_input_dim = num_actions * self.flattened_input_channels
        self.conv_num_channels = 16
        self.conv_input_dim = num_actions * self.conv_num_channels
        self.token_dimension = token_dimension
        self.num_embeddings = num_embeddings
        self.num_output_tokens = num_output_tokens
        self.num_input_tokens = num_input_tokens

        self.conv = nn.Sequential(
            nn.BatchNorm1d(self.flattened_input_channels),
            nn.Conv1d(self.flattened_input_channels, self.conv_num_channels, kernel_size=5, padding=2)
        )

        self.encoder_dim = self.num_input_tokens * self.token_dimension
        self.tokenizer = nn.Sequential(
            DoubleLayerMLP(self.conv_input_dim, self.encoder_dim, expansion_ratio=2)
        )

        self.k_input_proj = nn.Sequential(
            nn.LayerNorm(self.token_dimension),
            nn.Linear(self.token_dimension, self.token_dimension)
        )

        self.k_proj = nn.Sequential(
            nn.LayerNorm(self.token_dimension),
            nn.Linear(self.token_dimension, self.token_dimension)
        )

        self.v_input_proj = nn.Sequential(
            nn.LayerNorm(self.token_dimension),
            nn.Linear(self.token_dimension, self.token_dimension)
        )

        self.v_proj = nn.Sequential(
            nn.LayerNorm(self.token_dimension),
            nn.Linear(self.token_dimension, self.token_dimension)
        )

        self.embedding_matrix = nn.Parameter(torch.randn(self.token_dimension, self.num_embeddings))

        self.positional_encoding = nn.Parameter(torch.randn(self.num_output_tokens, self.token_dimension))

        self.mlp = DoubleLayerMLP(self.token_dimension, self.token_dimension, expansion_ratio=2)

        self.decoder_dim = self.num_output_tokens * self.token_dimension
        self.decoder = nn.Sequential(
            DoubleLayerMLP(self.decoder_dim, self.decoder_dim, expansion_ratio=2),
            DoubleLayerMLP(self.decoder_dim, self.decoder_dim, expansion_ratio=2),
            DoubleLayerMLP(self.decoder_dim, self.flattened_input_dim, expansion_ratio=2),
        )


        
    def forward(self, x : torch.Tensor):
        xc = self.conv(x.transpose(1, 2))

        xf = xc.flatten(1)
        input_tokens = self.tokenizer(xf).view(-1, self.num_input_tokens, self.token_dimension)

        k_cache = self.k_input_proj(input_tokens).transpose(1, 2)
        v_cache = self.v_input_proj(input_tokens)

        output_tokens = []
        for i in range(self.num_output_tokens):
            raw_next_token = self.positional_encoding[i].expand(x.shape[0], -1).unsqueeze(1)

            qkT = torch.matmul(raw_next_token, k_cache) / math.sqrt(self.token_dimension)
            softmax_scores = F.softmax(qkT, dim=2)
            attended_next_token = torch.matmul(softmax_scores, v_cache) + raw_next_token

            if i == 0:
                print(qkT)
                print(attended_next_token)
                print(self.positional_encoding[i])
                print(attended_next_token - raw_next_token)

            if i != self.num_output_tokens + 1:
                k_cache = torch.cat((k_cache, self.k_proj(attended_next_token).transpose(1, 2)), dim=2)
                v_cache = torch.cat((v_cache, self.v_proj(attended_next_token)), dim=1)

            mlp_next_token = self.mlp(attended_next_token.squeeze()).unsqueeze(1)

            embedding_probabilities = F.softmax(torch.matmul(mlp_next_token, self.embedding_matrix).view(-1, self.num_embeddings), dim=1)
            selected_index = torch.searchsorted(embedding_probabilities.cumsum(dim=1), torch.rand_like(embedding_probabilities[:, 0]).unsqueeze(1)).clamp(min=0, max=self.num_embeddings - 1)

            selected_token = self.embedding_matrix[:, selected_index].permute((1, 2, 0))

            # gradient passing trick
            selected_token = selected_token + mlp_next_token - mlp_next_token.detach()

            output_tokens.append(selected_token)

        output_tokens = torch.cat(output_tokens, dim=1).flatten(1)
        traj = self.decoder(output_tokens).view_as(x)

        return traj
    



model = TransformerAutoencoderMHA(num_actions=20, token_dimension=48, num_embeddings=96, num_input_tokens=6, num_output_tokens=8).to("cuda")
acds = ActionChunkDataset()
dataloader = torch.utils.data.DataLoader(
    acds,
    batch_size=256,
    shuffle=True
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)

num_epochs = 128
for i in range(num_epochs):

    print(f"Processing training for epoch {i}")

    model.train()
    acds.train_mode = True
    for j, sample in enumerate(dataloader):
        optimizer.zero_grad()

        sample = sample.to("cuda")

        output = model(sample)

        loss = torch.nn.functional.l1_loss(output, sample)
        loss.backward()

        optimizer.step()

        #print(f"\tL1 loss in btach {j} was {loss.item()}")

    print()

    print(f"Processing eval for epoch {i}")

    total_loss = 0.0
    num_eval_samples = 0

    model.eval()
    acds.train_mode = False
    for j, sample in enumerate(dataloader):
        optimizer.zero_grad()

        sample = sample.to("cuda")
        output = model(sample)

        loss = torch.nn.functional.mse_loss(output, sample)
        loss.backward()

        optimizer.step()

        #print(f"\tL2 loss in batch {j} was {loss.item()}")
        total_loss += loss.item() * output.shape[0]
        num_eval_samples += output.shape[0]

    print(f"Average L2 loss across all eval samples was {total_loss / num_eval_samples}")

    print("\n\n\n")


