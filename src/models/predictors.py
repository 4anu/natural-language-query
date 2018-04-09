import torch.nn as nn


class AggregatePredictor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self, args):
        super().__init__()
        self.args = args


class ConditionPredictor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self, args):
        super().__init__()
        self.args = args


class SelectPredictor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self, args):
        super().__init__()
        self.args = args
