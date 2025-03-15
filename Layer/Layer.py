from Function import Initializer
from enums import InitializerType
from Function.ActivationFunction import ActivationFunction
import torch

class Layer:
    """
    Layer Class.

    Consist of many neurons of Type Layer.
    """

    def __init__(self, weight_init : InitializerType, bias_init : InitializerType, param_1, param_2, num_neurons, input_size, activation = ActivationFunction.linear, layer_name = None):
        self.layer_name = layer_name
        self.activation_func = activation
        self.derivative_activation = None
        self.sum = None
        self.output = None

        # Initialize Neurons
        self.weights = Initializer.init_weights(
            weight_init=weight_init,
            param_1=param_1,
            param_2=param_2,
            input_size=input_size,
            num_neurons=num_neurons
        )
        self.biases = Initializer.init_bias(
            bias_init=bias_init,
            param_1=param_1,
            param_2=param_2,
            num_neurons=num_neurons
        )
        self.weights = torch.randn(num_neurons, input_size) * torch.sqrt(torch.tensor(2.0 / input_size))
        self.biases = torch.zeros(num_neurons)
        self.last_input = None

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
        self.last_input = x
        self.sum = torch.matmul(x, self.weights.T) + self.biases
        self.output = self.activation_func(self.sum)
        return self.output

    def backward(self, lr, delta_next):
        local_grad = self.derivative_activation(self.sum)
        delta = local_grad * delta_next

        grad_w = torch.matmul(delta.T, self.last_input) / self.last_input.size(0)
        grad_b = delta.mean(dim=0)

        self.weights -= lr * grad_w
        self.biases -= lr * grad_b

        delta_prev = torch.matmul(delta, self.weights)
        return delta_prev

    def __str__(self):
        return f"Layer Name: {self.layer_name}"
