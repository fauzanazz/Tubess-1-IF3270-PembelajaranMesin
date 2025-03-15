from Layer import Layer
from enums import InitializerType
from Function.ActivationFunction import ActivationFunction
import torch

class OutputLayer(Layer):
    """
    OutputLayer Class.

    Consist of many neurons of Output Layer
    """
    def __init__(self, weight_init : InitializerType, bias_init : InitializerType, input_size, num_neurons, param_1, param_2, activation = ActivationFunction.linear, layer_name = None):
        super().__init__(
            weight_init=weight_init,
            bias_init=bias_init,
            input_size=input_size,
            num_neurons=num_neurons,
            param_1=param_1,
            param_2=param_2,
            activation=activation,
            layer_name=layer_name
        )
        self.num_neurons = num_neurons

    def backward(self, lr, target):
        delta = self.output - target

        grad_w = torch.matmul(delta.T, self.last_input) / self.last_input.size(0)
        grad_b = delta.mean(dim=0)

        self.weights -= lr * grad_w
        self.biases -= lr * grad_b

        delta_prev = torch.matmul(delta, self.weights)
        return delta_prev

