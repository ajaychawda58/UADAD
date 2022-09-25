import numpy as np
from minisom import MiniSom
import torch
import numpy as np
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


def som_train(data, x=10, y=10, sigma=1, learning_rate= 0.05, iters= 10000):
    input_len = data.shape[1]
    print("SOM training started:")
    som = MiniSom(x= x, y= y, input_len=input_len, sigma=sigma, learning_rate=learning_rate)
    som.random_weights_init(data)
    som.train_random(data, iters)
    return som

def som_embedding_data(x_cat, x_cont, emb):
    emb_layer = EmbeddingLayer(emb)
    x_cat = emb_layer(x_cat)
    #i = 1
    #if i == 1:
    #    param = []
    #    for params in emb_layer.parameters():
    #        param.append(params.detach().numpy())
    #    with open('emb_weights.npy', 'wb') as f:
    #        np.save(f, param)
    data = torch.cat([x_cat, x_cont], -1)
    return data




def som_pred(som_model, data):
    model = som_model
    data = data.numpy()
    quantization_errors = np.linalg.norm(model.quantization(data) - data, axis=1)
    return quantization_errors