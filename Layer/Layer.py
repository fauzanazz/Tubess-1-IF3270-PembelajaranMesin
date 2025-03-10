from enums import InitializerType
from Neuron import Neuron
from Function.ActivationFunction import ActivationFunction
import torch

class Layer:
    """
    Layer Class.

    Consist of many neurons of Type Layer.
    """

    def __init__(self, weight_init : InitializerType, bias_init : InitializerType, input_size, output_size, param_1, param_2, activation = ActivationFunction.linear, layer_name = None):
        self.layer_name = layer_name
        self.neurons = [Neuron(weight_init, bias_init, input_size, param_1, param_2) for _ in range(output_size)]
        self.activation_func = activation
        self.sum = None
        self.output = None
        self.derivative_activation = None
        self.error_node = None

        match activation:
            case ActivationFunction.linear:
                self.derivative_activation = ActivationFunction.derivative_linear
            case ActivationFunction.relu:
                self.derivative_activation = ActivationFunction.derivative_relu
            case ActivationFunction.sigmoid:
                self.derivative_activation = ActivationFunction.derivative_sigmoid
            case ActivationFunction.tanh:
                self.derivative_activation = ActivationFunction.derivative_tanh
            case ActivationFunction.softmax:
                self.derivative_activation = ActivationFunction.derivative_softmax

    def forward(self, x):
        self.sum = torch.stack([neuron.forward(x) for neuron in self.neurons])
        self.output = self.activation_func(self.sum)
        return self.output

    def backward(self, lr, prev_layer):
        # Iterate through all neurons
        self.error_node = torch.zeros_like(self.output)
        for i, neuron in enumerate(self.neurons):
            # Calculate error node
            sum_of_weight = torch.sum(neuron.weights * prev_layer.error_node)
            error_node = self.output[i] * (1 - self.output[i]) * sum_of_weight
            neuron.error_node = error_node
            self.error_node[i] = error_node
            # Calculate cost weight
            self.neurons[i].cost_weight += -lr * error_node * self.output[i]
            # Calculate cost bias
            self.neurons[i].cost_bias += -lr * error_node * self.neurons[i].bias

    def update_weight(self):
        for neuron in self.neurons:
            neuron.weight_update()

    def update_bias(self):
        for neuron in self.neurons:
            neuron.bias_update()

    def __str__(self):
        return f"Layer Name: {self.layer_name}\nNeurons: {len(self.neurons)}\n"