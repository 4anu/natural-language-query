import torch
import torch.nn as nn
from torch.autograd import Variable

from models.embedding import EmbeddingLayer
from models.predictors import AggregatePredictor, SelectPredictor, ConditionPredictor


class NLQModel(nn.Module):
    def __init__(self, args, token_to_index, token_weights):
        super(NLQModel, self).__init__()
        self.args = args
        self.token_to_index = token_to_index

        self.aggregate_embedding_layer = EmbeddingLayer(emb_size=args.emb_size, gpu=args.gpu, token_to_index=token_to_index, token_weights=token_weights)
        # self.select_embedding_layer = EmbeddingLayer(emb_size=args.emb_size, gpu=args.gpu, token_to_index=token_to_index, token_weights=token_weights)
        # self.condition_embedding_layer = EmbeddingLayer(emb_size=args.emb_size, gpu=args.gpu, token_to_index=token_to_index, token_weights=token_weights)

        self.aggregate_predictor = AggregatePredictor(embedding_layer=self.aggregate_embedding_layer, args=args)
        # self.select_predictor = SelectPredictor(embedding_layer=self.select_embedding_layer, args=args)
        # self.condition_predictor = ConditionPredictor(embedding_layer=self.condition_embedding_layer, args=args)

        self.softmax = nn.Softmax()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.decay)

        if args.gpu:
            self.cuda()

    def forward(self, input):
        input = Variable(input.long())
        if self.args.gpu:
            input = input.cuda()

        return self.aggregate_predictor.forward(input)

    def calculate_loss(self, true_output, predicted_logits):
        true_output = Variable(torch.from_numpy(true_output))
        if self.args.gpu:
            true_output = true_output.cuda()

        return self.cross_entropy_loss(predicted_logits, true_output)

    def start_train(self, query_data_model, sql_data_model):
        num_batches = len(query_data_model)
        total_batches = self.args.epochs * num_batches
        for e in range(self.args.epochs):
            for b, (input, true_output) in enumerate(zip(query_data_model, sql_data_model)):
                self.aggregate_predictor.reset_hidden_state()
                true_output = Variable(true_output.long())
                if self.args.gpu:
                    true_output = true_output.cuda()
                logits = self.forward(input)
                loss = self.cross_entropy_loss(logits, true_output)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.args.gpu:
                    loss = loss.data.cuda().numpy()[0]
                else:
                    loss = loss.data.cpu().numpy()[0]
                print('{:d}/{:d} | Epoch {:d} | Loss: {:.2f}'.format(b * (e+1), total_batches, e+1, loss))

# TODO: Make all Predictors, forward function and loss.
