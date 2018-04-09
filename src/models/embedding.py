import torch.nn as nn
import constants.main_constants as const


class EmbeddingLayer(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self, embedding, emb_size=const.EMBEDDING_SIZE, gpu=False, train=False, token_to_index=None):
        super(EmbeddingLayer, self).__init__()
        self.embedding = embedding
        self.train = train
        self.gpu = gpu
        self.emb_size = emb_size
        self.token_to_index = token_to_index
