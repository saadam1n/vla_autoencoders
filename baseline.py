import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor

from data_stuff import ActionChunkDataset
from testing_framework import TrajectoryEncoder

class FastEncoder(TrajectoryEncoder):
    def __init__(self):
        super().__init__()

        self.tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

    def fit(self, acds, **kwargs):
        self.tokenizer = self.tokenizer.fit(acds.train_split())

    def encode(self, data):
        enc = self.tokenizer(data)

        avg_len = sum([len(tokens) for tokens in enc]) / len(enc)

        print(f"FastEncoder: average tokenized length was {avg_len}")

        return enc
    
    def decode(self, data):
        return torch.from_numpy(self.tokenizer.decode(data))