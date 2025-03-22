import torch

class ActionChunkDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(ActionChunkDataset, self).__init__()

        self.all_chunks = torch.load("data/traj_128k.pt", weights_only=True)
        self.all_chunks = self.all_chunks[torch.randperm(self.all_chunks.shape[0])]

        print(f"Traing data shape is {self.all_chunks.shape}")

        sorted_chunks, _ = torch.sort(self.all_chunks.flatten(0, 1), dim=0)

        lower = sorted_chunks[int(0.01 * sorted_chunks.shape[0])]
        upper = sorted_chunks[int(0.99 * sorted_chunks.shape[0])]
        range = upper - lower

        print("Dataset statistics:")
        print(f"\tlower: {lower}")
        print(f"\tupper: {upper}")
        print(f"\trange: {range}")

        self.all_chunks = (self.all_chunks - lower) / range

        self.train_mode = False

    def __getitem__(self, index):
        if self.train_mode:
            return self.all_chunks[index]
        else: 
            return self.all_chunks[-index]

    def __len__(self):
        if self.train_mode:
            return int(0.95 * self.all_chunks.shape[0])
        else:
            return int(0.05 * self.all_chunks.shape[0])
    