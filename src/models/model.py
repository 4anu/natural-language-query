import torch.nn as nn

from models.embedding import EmbeddingLayer
from models.predictors import AggregatePredictor, SelectPredictor, ConditionPredictor


class NLQModel(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self, embedding, args, token_to_index=None):
        super(NLQModel, self).__init__()
        self.embedding = embedding
        self.args = args
        self.token_to_index = token_to_index

        if args.train_embedding:
            self.aggregate_embedding_layer = EmbeddingLayer(embedding, emb_size=args.emb_size, gpu=args.gpu, train=True,
                                                            token_to_index=token_to_index)
            self.select_embedding_layer = EmbeddingLayer(embedding, emb_size=args.emb_size, gpu=args.gpu, train=True,
                                                         token_to_index=token_to_index)
            self.condition_embedding_layer = EmbeddingLayer(embedding, emb_size=args.emb_size, gpu=args.gpu, train=True,
                                                            token_to_index=token_to_index)
        else:
            self.embedding_layer = EmbeddingLayer(embedding, emb_size=args.emb_size, gpu=args.gpu)

        self.aggregate_predictor = AggregatePredictor(args=args)
        self.select_predictor = SelectPredictor(args=args)
        self.condition_predictor = ConditionPredictor(args=args)

        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()

        if args.gpu:
            self.cuda()
