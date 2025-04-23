import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dct import *
from testing_framework import TrajectoryEncoder
from data_stuff import ActionChunkDataset

import tqdm



# hash map is immutable, so we insert all chunks upon creation
class HashEncoder:
    def __init__(self, num_entries : int, quant_bins : int, xor_hash_primes : torch.Tensor, chunks : torch.Tensor):
        self.num_entries = num_entries
        self.xor_hash_primes = xor_hash_primes.detach().cpu().numpy()
        self.quant_bins = quant_bins

        self.mappings = [ [] for _ in range(self.num_entries) ]

        num_chunks = chunks.shape[0]
        np_chunks = chunks.detach().cpu().numpy()
        for i in tqdm.tqdm(range(num_chunks)):
            self.mappings[self.hash_chunk(np_chunks[i])].append(i)

    def hash_chunk(self, chunk : np.ndarray) -> int:
        return hash_dct(chunk, self.xor_hash_primes, self.quant_bins, self.num_entries)

    """
    Returns indices of which action chunks may map to this chunk
    """
    def fetch(self, chunk : torch.Tensor) -> list[int]:
        np_chunk = chunk.detach().cpu().numpy()

        return self.mappings[self.hash_chunk(np_chunk)]


class DoubleHashEncoder(TrajectoryEncoder):
    def __init__(self, num_entries : int, quant_bins : int):
        self.num_entries = num_entries
        self.quant_bins = quant_bins

    def fit(self, acds : ActionChunkDataset, **kwargs):
        self.chunks = acds.train_split()

        self.henc = [
            HashEncoder(num_entries=self.num_entries, quant_bins=self.quant_bins, xor_hash_primes=prime_sets[0], chunks=self.chunks),
            HashEncoder(num_entries=self.num_entries, quant_bins=self.quant_bins, xor_hash_primes=prime_sets[1], chunks=self.chunks),
        ]

    def encode(self, trajectories : torch.Tensor) -> int:
        indices = []
        for i in tqdm.tqdm(range(trajectories.shape[0])):
            traj = trajectories[i]

            candidates = sorted(self.henc[0].fetch(traj) + self.henc[1].fetch(traj))

            selected_index = -1
            for i in range(1, len(candidates)):
                if candidates[i] == candidates[i - 1]:
                    selected_index = candidates[i]
                    break

            if selected_index == -1 and len(candidates) > 0:
                selected_index = candidates[0]

            indices.append(selected_index)

        return indices

    def decode(self, indices : list[int]) -> torch.Tensor:
        return torch.stack(
            [
                self.chunks[idx] if idx != -1 else torch.zeros_like(self.chunks[0]) 
                for idx in indices
            ]
        )