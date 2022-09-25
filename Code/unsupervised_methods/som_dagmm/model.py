"""Implements all the components of the DAGMM model."""

from pandas import NA
import torch
import numpy as np
from torch import nn
import pickle
from minisom import MiniSom
from Code.classic_ML.SOM import som_train
from fastai.layers import Embedding
from fastai.torch_core import Module
from typing import List, Tuple


eps = torch.autograd.Variable(torch.FloatTensor([1.e-8]), requires_grad=False)

class EmbeddingLayer(Module):

    def __init__(self, emb_szs: List[Tuple[int, int]]):
        self.embeddings = torch.nn.ModuleList([Embedding(in_sz, out_sz) for in_sz, out_sz in emb_szs])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = [emb(x[..., i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(x, dim=-1)
        return x

class SOM_DAGMM(nn.Module):
    def __init__(self,dagmm, embedding, numerical, emb):
        super().__init__()
        self.dagmm = dagmm
        self.embedding = embedding
        self.numerical = numerical
        if self.embedding:
            self.emb_layer = EmbeddingLayer(emb)

    def forward(self, x_cat, x_cont, rec_data, save_path=None):
        if self.embedding:
            x_cat = self.emb_layer(x_cat)
        
        if self.numerical:
            x = x_cont
        else:
            x = torch.cat([x_cat, x_cont], -1)
        x = x.detach().numpy()
        with open(save_path, 'rb') as infile:
            som = pickle.load(infile)
        winners = [som.winner(i) for i in x]
        winners = torch.tensor([normalize_tuple(winners[i], 10) for i in range(len(winners))], dtype=torch.float32)
        return self.dagmm(x_cat, x_cont, rec_data, winners)



        
        
class DAGMM(nn.Module):
    def __init__(self, compression_module, estimation_module, gmm_module):
        """
        Args:
            compression_module (nn.Module): an autoencoder model that
                implements at leat a function `self.encoder` to get the
                encoding of a given input.
            estimation_module (nn.Module): a FFNN model that estimates the
                memebership of each input to a each mixture of a GMM.
            gmm_module (nn.Module): a GMM model that implements its mixtures
                as a list of Mixture classes. The GMM model should implement
                the function `self._update_mixtures_parameters`.
        """
        super().__init__()

        self.compressor = compression_module
        self.estimator = estimation_module
        self.gmm = gmm_module
        

    def forward(self, x_cat, x_cont, rec_data, winners):
        # Forward in the compression network.
        encoded = self.compressor.encode(x_cat, x_cont)
        decoded = self.compressor.decode(encoded)

        # Preparing the input for the estimation network.
        relative_ed = relative_euclidean_distance(rec_data, decoded)
        cosine_sim = cosine_similarity(rec_data, decoded)
        # Adding a dimension to prepare for concatenation.
        relative_ed = relative_ed.view(-1, 1)
        cosine_sim = relative_ed.view(-1, 1)
        latent_vectors = torch.cat([encoded, relative_ed, cosine_sim, winners], dim=1)
        # latent_vectors has shape [batch_size, dim_embedding + 2]

        # Updating the parameters of the mixture.
        if self.training:
            mixtures_affiliations = self.estimator(latent_vectors)
            # mixtures_affiliations has shape [batch_size, num_mixtures]
            self.gmm._update_mixtures_parameters(latent_vectors,
                                                 mixtures_affiliations)
        # Estimating the energy of the samples.
        return self.gmm(latent_vectors)


def relative_euclidean_distance(x1, x2, eps=eps):
    """x1 and x2 are assumed to be Variables or Tensors.
    They have shape [batch_size, dimension_embedding]"""
    num = torch.norm(x1 - x2, p=2, dim=1)  # dim [batch_size]
    denom = torch.norm(x1, p=2, dim=1)  # dim [batch_size]
    return num / torch.max(denom, eps)


def cosine_similarity(x1, x2, eps=eps):
    """x1 and x2 are assumed to be Variables or Tensors.
    They have shape [batch_size, dimension_embedding]"""
    dot_prod = torch.sum(x1 * x2, dim=1)  # dim [batch_size]
    dist_x1 = torch.norm(x1, p=2, dim=1)  # dim [batch_size]
    dist_x2 = torch.norm(x2, p=2, dim=1)  # dim [batch_size]
    return dot_prod / torch.max(dist_x1*dist_x2, eps)

def normalize_tuple(x, norm_val):
    a, b = x
    a = a/norm_val
    b = b/norm_val
    return (a,b)
