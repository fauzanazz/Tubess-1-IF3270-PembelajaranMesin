from Function.Initializer import Initializer
from enums import InitializerType
import torch

class Neuron:
    """
    Neuron Class.

    Consist of weights and bias for every input feature.
    """
    def __init__(self, weight_init : InitializerType, bias_init : InitializerType, input_size, param_1, param_2):
        self.weights = Initializer.init_weights(weight_init, param_1, param_2, input_size)
        self.bias = Initializer.init_bias(bias_init, param_1, param_2)
        self.cost_weight = torch.zeros(input_size)
        self.cost_bias = torch.zeros(1)
        self.error_node = None

    def forward(self, x):
        return torch.sum(x * self.weights) + self.bias

    def weight_update(self):
        self.weights += self.cost_weight
        self.cost_weight = torch.zeros_like(self.cost_weight)

    def bias_update(self):
        self.bias += self.cost_bias
        self.cost_bias = torch.zeros(1)