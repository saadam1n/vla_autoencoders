import torch

class ActionChunkDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(ActionChunkDataset, self).__init__()

        self.all_chunks = torch.load("data/traj_128k.pt", weights_only=True)
        self.all_chunks = self.all_chunks[torch.randperm(self.all_chunks.shape[0])]

        print(f"Traing data shape is {self.all_chunks.shape}")

        if True:
            sorted_chunks, _ = torch.sort(self.all_chunks.flatten(0, 1), dim=0)

            lower = sorted_chunks[int(0.01 * sorted_chunks.shape[0])]
            upper = sorted_chunks[int(0.99 * sorted_chunks.shape[0])]
            range = upper - lower

            print("Dataset statistics:")
            print(f"\tlower: {lower}")
            print(f"\tupper: {upper}")
            print(f"\trange: {range}")

            self.all_chunks = (self.all_chunks - lower) / range
            self.all_chunks = 2 * self.all_chunks - 1
            self.all_chunks = torch.clamp(self.all_chunks, -1, 1)
        else:
            flattened_chunks = self.all_chunks.flatten(0, 1)
            var, mean = torch.var_mean(flattened_chunks, dim=0)

            std = torch.sqrt(var + 1e-5)



            print("Dataset statistics:")
            print(f"\tmean: {mean}")
            print(f"\tvar : {var}")
            print(f"\tstd : {std}")

            mean = mean.view(1, 1, -1)
            std = std.view(1, 1, -1)

            self.all_chunks = (self.all_chunks - mean) / std

            check_var, check_mean = torch.var_mean(self.all_chunks.flatten(0, 1), dim=0)

            print("Check stats:")
            print(f"\tmean: {check_mean}")
            print(f"\tvar : {check_var}")




        self.train_mode = False

        self.train_size = int(0.95 * self.all_chunks.shape[0])
        self.test_size = self.all_chunks.shape[0] - self.train_size

    def __getitem__(self, index):
        if self.train_mode:
            return self.all_chunks[index]
        else: 
            return self.all_chunks[-index]

    def __len__(self):
        if self.train_mode:
            return self.train_size
        else:
            return self.test_size
        
    def train_split(self):
        return self.all_chunks[:self.train_size]

    def test_split(self):
        return self.all_chunks[self.train_size:]
    