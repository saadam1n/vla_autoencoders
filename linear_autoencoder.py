import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_stuff import ActionChunkDataset
from transformers import AutoProcessor

class FiniteScalarQuantization(nn.Module):
    def __init__(self, num_bins):
        super(FiniteScalarQuantization, self).__init__()

        self.quantization_granularity = num_bins

    def forward(self, x: torch.Tensor):
        z = x.tanh() * self.quantization_granularity / 2.0

        zhat = z + (z.round() - z).detach()

        return zhat


class LinearAutoencoder(nn.Module):
    """
    time_horizon - number of actions within action chunk
    action_dim - number of dims per action
    """
    def __init__(self, time_horizon, action_dim, vocab_size, num_tokens):
        super(LinearAutoencoder, self).__init__()

        self.action_dim = action_dim
        self.time_horizon = time_horizon
        self.vocab_size = vocab_size
        self.num_tokens = num_tokens
        self.total_chunk_size = self.action_dim * self.time_horizon

        # short for conv expansion ration

        self.encoder_ffn = nn.Sequential(
            nn.BatchNorm1d(self.total_chunk_size),
            nn.Linear(self.total_chunk_size, 8 * self.total_chunk_size),
            nn.ReLU(),
            #nn.Dropout(),
            nn.BatchNorm1d(8 * self.total_chunk_size),
            nn.Linear(8 * self.total_chunk_size, self.num_tokens) # switch output rows/bias
        )


        self.encoder_skip = nn.Sequential(
            nn.BatchNorm1d(self.total_chunk_size),
            nn.Linear(self.total_chunk_size, self.num_tokens), # switch output rows/bias
        )


        self.fsq = nn.Sequential(
            nn.BatchNorm1d(self.num_tokens), # switch running_mean and var
            FiniteScalarQuantization(self.vocab_size),
        )

        self.decoder_ffn = nn.Sequential(
            nn.BatchNorm1d(self.num_tokens), # switch running_mean and var
            nn.Linear(self.num_tokens, 8 * self.total_chunk_size), # switch columns only
            nn.ReLU(),
            #nn.Dropout(),
            nn.BatchNorm1d(8 * self.total_chunk_size),
            nn.Linear(8 * self.total_chunk_size, self.total_chunk_size)
        )

        self.decoder_skip = nn.Sequential(
            nn.BatchNorm1d(self.num_tokens), # switch running_mean and var
            nn.Linear(self.num_tokens, self.total_chunk_size) # switch columns only
        )

    def forward(self, x : torch.Tensor):
        # x is (N, L, C) format
        # where L = time_horizon and C = action_dim
        # we want (N, C, L) format (before we inevitably flatten it)

        xq = self.encode(x)

        xd = self.decode(xq)

        return xd.view_as(x)
    
    def encode(self, x : torch.Tensor):
        xf = x.flatten(1)

        xe = self.encoder_ffn(xf) + self.encoder_skip(xf)

        xq = self.fsq(xe)

        return xq
    
    def decode(self, xq : torch.Tensor):
        xd = self.decoder_ffn(xq)# + self.decoder_skip(xq)

        return xd
    
    def transpose(self, indices : torch.Tensor):
        print(f"TRANSPOSITON SHAPE {indices.shape}")
        print(f"\tVAL{indices}")

        self.encoder_ffn[4].weight.data = self.encoder_ffn[4].weight.data[indices, :]
        self.encoder_ffn[4].bias.data = self.encoder_ffn[4].bias.data[indices]

        self.encoder_skip[1].weight.data = self.encoder_skip[1].weight.data[indices, :]
        self.encoder_skip[1].bias.data = self.encoder_skip[1].bias.data[indices]

        self.fsq[0].running_mean.data = self.fsq[0].running_mean.data[indices]
        self.fsq[0].running_var.data = self.fsq[0].running_var.data[indices]
        self.fsq[0].weight.data = self.fsq[0].weight.data[indices]
        self.fsq[0].bias.data = self.fsq[0].bias.data[indices]
    
        self.decoder_ffn[0].running_mean.data = self.decoder_ffn[0].running_mean.data[indices]
        self.decoder_ffn[0].running_var.data = self.decoder_ffn[0].running_var.data[indices]
        self.decoder_ffn[0].weight.data = self.decoder_ffn[0].weight.data[indices]
        self.decoder_ffn[0].bias.data = self.decoder_ffn[0].bias.data[indices]
        self.decoder_ffn[1].weight.data = self.decoder_ffn[1].weight.data[:, indices]

        self.decoder_skip[0].running_mean.data = self.decoder_skip[0].running_mean.data[indices]
        self.decoder_skip[0].running_var.data = self.decoder_skip[0].running_var.data[indices]
        self.decoder_skip[0].weight.data = self.decoder_skip[0].weight.data[indices]
        self.decoder_skip[0].bias.data = self.decoder_skip[0].bias.data[indices]
        self.decoder_skip[1].weight.data = self.decoder_skip[1].weight.data[:, indices]


acds = ActionChunkDataset()
dataloader = torch.utils.data.DataLoader(
    acds,
    batch_size=256,
    shuffle=True
)

f = open("data/results.csv", "w")
f.write("vocab_size,\tnum_tokens,\tMSE_loss,\tbpe_num_tokens,\tbpe_MSE_loss\n")
f.flush()
for vocab_size in [16, 64, 256, 1024]:
    for num_tokens in [4, 8, 12, 16, 24]:
        print(f"PROCESSING CONFIG VOCAB_SIZE={vocab_size}, NUM_TOKENS={num_tokens}")

        model = LinearAutoencoder(time_horizon=20, action_dim=7, vocab_size=vocab_size, num_tokens=num_tokens).to("cuda")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)

        num_epochs = 128
        for i in range(num_epochs):

            print(f"Processing training for epoch {i}")

            total_train_loss = 0.0
            num_train_samples = 0

            model.train()
            acds.train_mode = True
            for j, sample in enumerate(dataloader):
                optimizer.zero_grad()

                sample = sample.to("cuda")

                output = model(sample)

                loss = torch.nn.functional.l1_loss(output, sample)
                loss.backward()

                optimizer.step()

                total_train_loss += loss.item() * output.shape[0]
                num_train_samples += output.shape[0]
                #print(f"\tL1 loss in batch {j} was {loss.item()}")

            avg_l1 = total_train_loss / num_train_samples

            print(f"Average L1 loss across all train samples was {avg_l1}")

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

                loss = F.mse_loss(output, sample)
                loss.backward()

                optimizer.step()

                #print(f"\tL2 loss in batch {j} was {loss.item()}")
                total_loss += loss.item() * output.shape[0]
                num_eval_samples += output.shape[0]

            avg_l2 = total_loss / num_eval_samples

            print(f"Average L2 loss across all eval samples was {avg_l2}")

            print("\n\n\n")

        model.eval()

        all_input = acds.all_chunks.to("cuda")

        # neuron transposition
        expected_out = model(all_input)
        dropout_check = model(all_input)

        print(f"Dropout check was {F.mse_loss(dropout_check, expected_out)}")

        expected_fsq = model.encode(all_input)

        def get_value_grad():
            compressed_coeffs = model.encode(all_input)
            compressed_coeffs.retain_grad()

            traj = model.decode(compressed_coeffs).view_as(all_input)

            fake_loss = torch.nn.functional.mse_loss(traj, torch.zeros_like(traj)).backward()

            return compressed_coeffs

        value_grad = get_value_grad().grad.abs().sum(dim=0)
        print(value_grad)

        _, indices = torch.sort(value_grad, descending=True)
        indices = indices.flatten()

        with torch.no_grad():
            model.transpose(indices)

        actual_out = model(all_input)
        actual_fsq = model.encode(all_input)

        print(f"FSQ MSE loss from transpositon was {F.mse_loss(actual_fsq, expected_fsq[:, indices])}")
        print(f"TOT MSE loss from transpositon was {F.mse_loss(actual_out, expected_out)}")
        print(actual_fsq[0])
        print(expected_fsq[0, indices])
        print(expected_fsq[0])

        # tokenize entire dataset
        # move tokens [-1, 1]
        xq = (model.encode(all_input) * 2.0 / model.vocab_size).clamp(min=-1, max=1)

        # (N, C) -> (N, 1, C) to put everything in one DCT channel
        xq_np = xq.unsqueeze(1).detach().cpu().numpy()

        tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True).fit(xq_np)

        xq_tokenized = tokenizer(xq_np)

        avg_seq_len = sum([len(tokens) for tokens in xq_tokenized]) / len(xq_tokenized)

        xd = model.decode(
            torch.from_numpy(tokenizer.decode(xq_tokenized)).to("cuda").squeeze(1).float() * model.vocab_size / 2.0
        ).view_as(all_input)

        bpe_avg_l2 = F.mse_loss(xd, all_input.to("cuda"))

        print(f"BPE len was {avg_seq_len} and loss was {bpe_avg_l2}")

        f.write(f"{vocab_size},\t{num_tokens},\t{avg_l2},\t{avg_seq_len},\t{bpe_avg_l2}\n")
        f.flush()

f.close()

print(f"REMEMBER TO RENAME THE FILE!!!! THIS TEST WAS RAN WITH DROPOUT!!!")
        
