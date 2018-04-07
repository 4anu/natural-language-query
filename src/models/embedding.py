import torch
import torch.nn as nn
import constants.main_constants as const


class Embedding(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self, glove, emb_size=const.EMBEDDING_SIZE, gpu=False, train=False):
        super(Embedding, self).__init__()
        self.glove = glove
        self.train = train
        self.gpu = gpu
        self.emb_size = emb_size
