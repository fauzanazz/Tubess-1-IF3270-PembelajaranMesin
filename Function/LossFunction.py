import torch

class LossFunction:
    """
    LossFunction Class.

    This class defines loss functions and their derivatives.
    """

    @staticmethod
    @torch.compile
    def mean_squared_error(y_pred, y_true, epsilon=1e-7):
        y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)
        return torch.mean((y_pred - y_true) ** 2)

    @staticmethod
    def mean_squared_error_derivative(y_pred, y_true, epsilon=1e-7):
        y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)
        return 2 * (y_pred - y_true) / y_true.size(0)

    @staticmethod
    @torch.compile
    def binary_cross_entropy(y_pred, y_true, epsilon=1e-7):
        y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)
        return -(y_pred.log()*y_true + (1-y_true)*(1-y_pred).log()).mean()

    @staticmethod
    def binary_cross_entropy_derivative(y_pred, y_true, epsilon=1e-7):
        y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)
        return ((y_pred - y_true) / (y_pred * (1 - y_pred))) / y_true.size(0)

    @staticmethod
    @torch.compile
    def categorical_cross_entropy(y_true, y_pred, epsilon=1e-7):
        y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)
        return (-torch.sum(y_true * torch.log(y_pred)).mean())/2


    @staticmethod
    def categorical_cross_entropy_derivative(y_pred, y_true, epsilon=1e-7):
        y_pred = torch.clamp(y_pred, epsilon, 1.0)
        return -(y_true / y_pred) / y_true.size(0)


if __name__ == "__main__":
    y_true = torch.tensor([1, 0, 0, 1, 1], dtype=torch.float32)
    y_pred = torch.tensor([0.9, 0.1, 0.2, 0.8, 0.9], dtype=torch.float32)


    print("Loss Functions MSE")
    print(LossFunction.mean_squared_error(y_pred, y_true))
    print(torch.nn.functional.mse_loss(y_pred, y_true))

    print("Loss Functions BCE")
    print(LossFunction.binary_cross_entropy(y_pred, y_true))
    print(torch.nn.functional.binary_cross_entropy(y_pred, y_true))

    print("Loss Functions CCE")
    print(LossFunction.categorical_cross_entropy(y_pred, y_true))
    print(torch.nn.functional.cross_entropy(y_pred, y_true))
