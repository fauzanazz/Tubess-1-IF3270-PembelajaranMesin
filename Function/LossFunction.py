import torch

class LossFunction:
    """
    LossFunction Class.

    This class is used to define the loss function.
    """

    @staticmethod
    def mean_squared_error(y_pred, y_true):
        return torch.mean((y_pred - y_true) ** 2)

    @staticmethod
    def binary_cross_entropy(y_pred, y_true):
        return -torch.sum(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

    @staticmethod
    def categorical_cross_entropy(y_pred, y_true):
        return -torch.sum(y_true * torch.log(y_pred))