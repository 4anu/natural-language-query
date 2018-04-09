import numpy as np
import torch
import torch.nn as nn
import constants.main_constants as const


class EmbeddingLayer(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self, embedding, emb_size=const.EMBEDDING_SIZE, gpu=False, train=False, token_to_index=None, token_weights=None):
        super(EmbeddingLayer, self).__init__()
        self.embedding = embedding
        self.train = train
        self.gpu = gpu
        self.emb_size = emb_size
        self.token_to_index = token_to_index

        if train:
            self.total_embeddings = len(token_to_index)
            self.embedding_layer = nn.Embedding(self.total_embeddings, emb_size)
            self.embedding_layer.weight = nn.Parameter(torch.from_numpy(token_weights.astype(np.float32)))

# TODO: Make embedding batch functions
