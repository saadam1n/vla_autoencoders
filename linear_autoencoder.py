import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_stuff import ActionChunkDataset
from transformers import AutoProcessor

class FeedForwardSwiGLU(nn.Module):
    """Some Information about FeedForwardSwiGLU"""
    def __init__(self, features_in, features_out, expansion_ratio):
        super(FeedForwardSwiGLU, self).__init__()

        features_latent = expansion_ratio * max(features_in, features_out)

        self.swish_linear = nn.Sequential(
            nn.BatchNorm1d(features_in),
            nn.Linear(features_in, features_latent)
        )

        self.swish_beta = nn.Parameter(torch.randn(features_latent))

        self.gated_linear = nn.Sequential(
            nn.BatchNorm1d(features_in),
            nn.Linear(features_in, features_latent)
        )

        self.output_linear = nn.Sequential(
            nn.BatchNorm1d(features_latent),
            nn.Linear(features_latent, features_out)
        )

        self.skip_linear = nn.Sequential(
            nn.BatchNorm1d(features_in),
            nn.Linear(features_in, features_out)
        ) if features_in != features_out else None


    def forward(self, x : torch.Tensor):

        swin = self.swish_linear(x)
        swish = swin * F.sigmoid(self.swish_beta * swin)

        gated = self.gated_linear(x)

        skip = x if self.skip_linear is None else self.skip_linear(x)
        output = self.output_linear(gated * swish) + skip

        return output

class RoundLU(nn.Module):
    """Some Information about RoundLU"""
    def __init__(self, num_features, vocab_size):
        super(RoundLU, self).__init__()\
        
        print("Utilizing RoundLU module!")

        self.act_mid = num_features // 2
        self.fsq = FiniteScalarQuantization(vocab_size)

    def forward(self, x):

        xrelu = F.relu(x[:, :self.act_mid])
        xfsq = self.fsq(x[:, self.act_mid:])

        xact = torch.cat((xrelu, xfsq), dim=1)

        return xact


class FeedForwardReLU(nn.Module):
    """Some Information about FeedForwardReLU"""
    def __init__(self, features_in, features_out, expansion_ratio, roundlu_vocab = None):
        super(FeedForwardReLU, self).__init__()

        features_latent = expansion_ratio * max(features_in, features_out)

        self.feed_forward = nn.Sequential(
            nn.BatchNorm1d(features_in),
            nn.Linear(features_in, features_latent),
            nn.ReLU() if roundlu_vocab is None else RoundLU(features_latent, roundlu_vocab),
            nn.Linear(features_latent, features_out)
        )


        self.skip_linear = nn.Sequential(
            nn.BatchNorm1d(features_in),
            nn.Linear(features_in, features_out)
        ) if features_in != features_out else None

    def forward(self, x):
        skip = x if self.skip_linear is None else self.skip_linear(x)
        output = self.feed_forward(x) + skip

        return output


class FiniteScalarQuantization(nn.Module):
    def __init__(self, num_bins):
        super(FiniteScalarQuantization, self).__init__()

        self.quantization_granularity = num_bins - 1

    def forward(self, x: torch.Tensor):
        z = x.tanh() * self.quantization_granularity

        zhat = z + (z.round() - z).detach()

        return zhat


class LinearAutoencoder(nn.Module):
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

    def forward(self, x : torch.Tensor):
        # x is (N, L, C) format
        # where L = time_horizon and C = action_dim
        # we want (N, C, L) format (before we inevitably flatten it)

        xq = self.encode(x)

        xd = self.decode(xq)

        return xd.view_as(x).cumsum(dim=1)
    
    def encode(self, x : torch.Tensor) -> torch.Tensor:
        # take consecutive differences
        xr = torch.cat((x[:, :1, :], torch.diff(x, dim=1)), dim=1)

        xf = xr.flatten(1)

        xe = self.encoder(xf)

        xq = self.fsq(xe)

        return xq
    
    def decode(self, xq : torch.Tensor) -> torch.Tensor:
        xd = self.decoder(xq)

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


acds = ActionChunkDataset()
dataloader = torch.utils.data.DataLoader(
    acds,
    batch_size=512,
    shuffle=True
)

f = open("data/results.csv", "w")
f.write("vocab_size,\tnum_tokens,\tMSE_loss,\tbpe_num_tokens,\tbpe_MSE_loss\n")
f.flush()
for vocab_size in [16, 64, 256, 1024]:
    for num_tokens in [4, 8, 12, 16, 24]:
        print(f"PROCESSING CONFIG VOCAB_SIZE={vocab_size}, NUM_TOKENS={num_tokens}")

        model = LinearAutoencoder(time_horizon=20, action_dim=7, vocab_size=vocab_size, num_tokens=num_tokens).to("cuda")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.9)

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

        # (N, C) -> (N, C, 1) to put everything in one DCT channel
        xq_np = xq.unsqueeze(2).detach().cpu().numpy()

        tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True).fit(xq_np)

        xq_tokenized = tokenizer(xq_np)

        avg_seq_len = sum([len(tokens) for tokens in xq_tokenized]) / len(xq_tokenized)

        xd = model.decode(
            torch.from_numpy(tokenizer.decode(xq_tokenized)).to("cuda").squeeze(2).float() * model.vocab_size / 2.0
        ).view_as(all_input)

        bpe_avg_l2 = F.mse_loss(xd, all_input.to("cuda"))

        print(f"BPE len was {avg_seq_len} and loss was {bpe_avg_l2}")

        f.write(f"{vocab_size},\t{num_tokens},\t{avg_l2},\t{avg_seq_len},\t{bpe_avg_l2}\n")
        f.flush()

f.close()

print(f"REMEMBER TO RENAME THE FILE!!!! THIS TEST WAS RAN WITH DROPOUT!!!")
        
