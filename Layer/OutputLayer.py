from Layer import Layer
from enums import InitializerType
from Function.ActivationFunction import ActivationFunction
import torch
from Utils import Utils

class OutputLayer(Layer):
    """
    OutputLayer Class.

    Consist of many neurons of Output Layer
    """
    def __init__(self, weight_init : InitializerType, bias_init : InitializerType, input_size, output_size, param_1, param_2, activation = ActivationFunction.linear, layer_name = None):
        super().__init__(weight_init, bias_init, input_size, output_size, param_1, param_2, activation, layer_name)

    def forward(self, x):
        self.sum = torch.stack([neuron.forward(x) for neuron in self.neurons])
        self.output = ActivationFunction.sigmoid(self.sum)
        return self.output

    def backward(self, lr, target = None, layer = None):
        self.error_node = torch.zeros_like(self.output)
        target_min_output = Utils.output_minus_target(self.output, target)
        for i, neuron in enumerate(self.neurons):
            target_delta = target_min_output[i]
            error_node = self.output[i] * (1 - self.output[i]) * target_delta
            self.error_node[i] = error_node
            neuron.error_node = error_node
            self.neurons[i].cost_weight += -lr * error_node * self.output[i]
            self.neurons[i].cost_bias += -lr * error_node

