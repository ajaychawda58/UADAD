"""Defines the compression network."""

import torch
from torch import nn
from fastai.layers import Embedding
from fastai.torch_core import Module
from typing import List, Tuple
class EmbeddingLayer(Module):

    def __init__(self, emb_szs: List[Tuple[int, int]]):
        self.embeddings = torch.nn.ModuleList([Embedding(in_sz, out_sz) for in_sz, out_sz in emb_szs])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = [emb(x[..., i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(x, dim=-1)
        return x

class CompressionNetwork(nn.Module ):
    """Defines a compression network."""
    def __init__(self, embedding, numerical, input_dim, output_dim, emb_szs, latent_dim):
        super().__init__()
        self.embedding = embedding
        self.numerical = numerical
        if self.embedding:
            self.emb_layer = EmbeddingLayer(emb_szs)
        self.encoder = nn.Sequential(nn.Linear(input_dim, 10),
                                     nn.Tanh(),
                                     nn.Linear(10, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 10),
                                     nn.Tanh(),
                                     nn.Linear(10, output_dim))

        self._reconstruction_loss = nn.MSELoss()

    def forward(self, x_cat, x_cont):
        if self.embedding:
            x_cat = self.emb_layer(x_cat)
        
        if self.numerical:
            x = x_cont
        else:
            x = torch.cat([x_cat, x_cont], -1)
            
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

    def encode(self, x_cat, x_cont):        
        if self.numerical:
            x = x_cont
        else:
            x = torch.cat([x_cat, x_cont], -1)
        return self.encoder(x)

    def decode(self, input):
        return self.decoder(input)

    def reconstruction_loss(self, x_cat, x_cont, rec_data):
        if self.embedding:
            x_cat = self.emb_layer(x_cat)        
        if self.numerical:
            x = x_cont
        else:
            x = torch.cat([x_cat, x_cont], -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return self._reconstruction_loss(decoded, rec_data)