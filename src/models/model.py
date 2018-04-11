import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import accuracy_score

from models.embedding import EmbeddingLayer
from models.predictors import AggregatePredictor, SelectPredictor, ConditionPredictor
import constants.main_constants as const


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

    def start_train(self, query_data_model, sql_data_model):
        num_batches = len(query_data_model)
        total_batches = self.args.epochs * num_batches
        try:
            for e in range(self.args.epochs):
                epoch_accuracy = 0
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
                        loss = loss.data.cuda().cpu()[0]
                        logits = logits.data.cuda().cpu()
                        true_output = true_output.data.cuda().cpu()
                    else:
                        loss = loss.data.cpu().numpy()[0]
                        logits = logits.data.cpu().numpy()
                        true_output = true_output.data.cpu().numpy()
                    predicted_output = np.argmax(logits, 1)
                    accuracy = 100 * accuracy_score(predicted_output, true_output)
                    epoch_accuracy += accuracy * self.args.batch_size
                    print('{:d}/{:d} | Epoch {:d} | Loss: {:.3f} | Accuracy: {:.2f}'.format(b * (e+1), total_batches, e+1, loss, accuracy))
                if (e+1) % 5 == 0:
                    torch.save(self.aggregate_embedding_layer.state_dict(), const.AGG_EMB_SAVE_MODEL.format(e+1))
                    torch.save(self.aggregate_predictor.state_dict(), const.AGG_SAVE_MODEL.format(e+1))
                print('Epoch {:d} finished with Accuracy: {:.2f}'.format(e+1, epoch_accuracy/num_batches))
                # TODO: Save model based on dev set accuracy.
        except KeyboardInterrupt:
            print('-' * 100)
            print('Exiting from training..')

# TODO: Make all Predictors, forward function and loss.
