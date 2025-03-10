from functools import lru_cache
import torch

class ActivationFunction:
    """
    ActivationFunction Class.

    This class is used to define the activation function.
    """

    @staticmethod
    @lru_cache(maxsize=None)
    def linear(x):
        return x

    @staticmethod
    @lru_cache(maxsize=None)
    def relu(x):
        return torch.relu(x)

    @staticmethod
    @lru_cache(maxsize=None)
    def sigmoid(x):
        return torch.sigmoid(x)

    @staticmethod
    @lru_cache(maxsize=None)
    def tanh(x):
        return torch.tanh(x)

    @staticmethod
    @lru_cache(maxsize=None)
    def softmax(x):
        return torch.softmax(x, dim=0)

    @staticmethod
    def derivative_linear(_):
        return 1

    @staticmethod
    def derivative_relu(x):
        return torch.where(x > 0, 1, 0)

    @staticmethod
    def derivative_sigmoid(x):
        return torch.sigmoid(x) * (1 - torch.sigmoid(x))

    @staticmethod
    def derivative_tanh(x):
        return (2 / torch.exp(x) - torch.exp(-x)) ** 2

    @staticmethod
    def derivative_softmax(x):
        s = torch.softmax(x, dim=0).reshape(-1, 1)
        return torch.diagflat(s) - torch.mm(s, s.T)