import torch
from torch import nn
import torch.nn.functional as F
from fastai.layers import Embedding
from fastai.torch_core import Module
from typing import List, Tuple

class RSRLayer(nn.Module):
    def __init__(self, d, D):
        super().__init__()
        self.d = d
        self.D = D
        self.A = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(d, D)))

    def forward(self, z):
        x = self.A @ z.view(z.size(0), self.D, 1)
        return x.squeeze(2)

class RSRLoss(nn.Module):
    def __init__(self, L1, L2, d, D):
        super().__init__()
        self.L1 = L1
        self.L2 = L2
        self.d = d
        self.D = D
        self.register_buffer("Id", torch.eye(d))
    
    def forward(self, z, A):
        z_hat = A @ z.view(z.size(0), self.D, 1)
        AtAz = (A.T @ z_hat).squeeze(2)
        term1 = torch.sum(torch.norm(z -AtAz, p=2))
        term2 = torch.sum(torch.norm(A@A.T - self.Id, p=2))**2
        return self.L1 * term1 + self.L2 * term2

class EmbeddingLayer(Module):

    def __init__(self, emb_szs: List[Tuple[int, int]]):
        self.embeddings = torch.nn.ModuleList([Embedding(in_sz, out_sz) for in_sz, out_sz in emb_szs])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = [emb(x[..., i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(x, dim=-1)
        return x

class RSRAE(nn.Module):
    def __init__(self,embedding, numerical, input_dim, output_dim, emb_szs, d, D):
        super().__init__()
        self.embedding = embedding
        self.numerical = numerical
        if self.embedding:
            self.emb_layer = EmbeddingLayer(emb_szs)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, D)
        )
        self.rsr = RSRLayer(d, D)

        self.decoder = nn.Sequential(
            nn.Linear(d, D),
            nn.LeakyReLU(),
            nn.Linear(D, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, output_dim)
        )
    def forward(self, x_cat, x_cont):
        if self.embedding:
            x_cat = self.emb_layer(x_cat)
        
        if self.numerical:
            x = x_cont
        else:
            x = torch.cat([x_cat, x_cont], -1)
           
        encoded = self.encoder(x)
        latent = self.rsr(encoded)
        decoded = self.decoder(F.normalize(latent, p=2))
        return encoded, decoded, latent, self.rsr.A

    def L21(self, y_hat, y):
        return torch.sum(torch.pow(torch.norm(y - y_hat, p=2), 1))
