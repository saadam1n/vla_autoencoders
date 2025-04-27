import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

class FeedForwardGELU(nn.Module):
    """Some Information about FeedForwardReLU"""
    def __init__(self, features_in, features_out, expansion_ratio):
        super(FeedForwardGELU, self).__init__()

        features_latent = expansion_ratio * max(features_in, features_out)

        self.feed_forward = nn.Sequential(
            nn.BatchNorm1d(features_in),
            nn.Linear(features_in, features_latent),
            nn.GELU() ,
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

class FeedForwardReLUInjectable(nn.Module):
    """Some Information about FeedForwardReLU"""
    def __init__(self, features_in, features_out, expansion_ratio, emb_dim = 0):
        super(FeedForwardReLUInjectable, self).__init__()

        features_latent = expansion_ratio * max(features_in, features_out)

        self.feed_forward = nn.Sequential(
            nn.BatchNorm1d(features_in),
            nn.Linear(features_in, features_latent),
        )

        self.emb_linear = nn.Linear(emb_dim, features_latent)

        self.ffn_2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(features_latent, features_out)
        )

        self.skip_linear = nn.Sequential(
            nn.BatchNorm1d(features_in),
            nn.Linear(features_in, features_out)
        ) if features_in != features_out else None

        self.emb_skip = nn.Linear(emb_dim, features_out)

    def forward(self, x, emb):
        t_emb = self.emb_linear(emb)
        s_emb = self.emb_skip(emb)

        skip = x if self.skip_linear is None else self.skip_linear(x)
        output = self.ffn_2(self.feed_forward(x) + t_emb) + skip + s_emb

        return output

class FiniteScalarQuantization(nn.Module):
    def __init__(self, num_bins):
        super(FiniteScalarQuantization, self).__init__()

        self.quantization_granularity = num_bins / 2

    def forward(self, x: torch.Tensor):
        z = x.tanh() * self.quantization_granularity

        zhat = z + (z.round() - z).detach()

        return zhat
    

class VQVAEBlock(nn.Module):
    """Some Information about VQVAEBlock"""
    def __init__(self, num_embeddings, embedding_dim):
        super(VQVAEBlock, self).__init__()

        self.embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.beta = 0.25

    """
    Returns output and encoder loss
    """
    def forward(self, x : torch.Tensor):

        # x is (N, L, C) tensor
        # we want to get (N, L, E) distances and then select argmin
        # we will need to use our good old friend broadcasting

        # (N, L, C) -> (N, L, 1, C)
        # (N, L, 1, C) - (E, C) = (N, L, E, C)
        # (N, L, E, C) -> (N, L, E)
        offsets = x.unsqueeze(2) - self.embeddings.weight
        indices = (offsets ** 2).sum(dim=3).argmin(dim=2, keepdim=True)

        # now a (N, C) vector
        # detach so gradient doesn't flow to embeddings
        quant = self.embeddings(indices).squeeze(2)

        # compute loss
        # essentially these losses force 
        # due to the beta term we have to use two calls to mse_loss
        vqloss = F.mse_loss(quant, x.detach()) + self.beta * F.mse_loss(x, quant.detach())

        # apply stop gradient to make loss w.r.t. VQ VAE flow directly to encoder
        sgquant = quant.detach() + x - x.detach()

        return sgquant, vqloss