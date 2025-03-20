from Function import Initializer
from enums import InitializerType
from Function.ActivationFunction import ActivationFunction
import numpy as np

class Layer:
    def __init__(self, weight_init: InitializerType, bias_init: InitializerType, 
                param_1, param_2, num_neurons, input_size, 
                activation=ActivationFunction.linear, layer_name=None):
        self.layer_name = layer_name
        self.activation_func = activation
        self.derivative_activation = None
        self.sum = None
        self.output = None

        # Initialize weights and biases
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
        
        # Override with Xavier initialization for consistency
        self.weights = np.random.randn(num_neurons, input_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros(num_neurons)
        self.last_input = None

        # Set the derivative function based on activation
        if activation == ActivationFunction.linear:
            self.derivative_activation = ActivationFunction.derivative_linear
        elif activation == ActivationFunction.relu:
            self.derivative_activation = ActivationFunction.derivative_relu
        elif activation == ActivationFunction.sigmoid:
            self.derivative_activation = ActivationFunction.derivative_sigmoid
        elif activation == ActivationFunction.tanh:
            self.derivative_activation = ActivationFunction.derivative_tanh
        elif activation == ActivationFunction.softmax:
            self.derivative_activation = ActivationFunction.derivative_softmax

    def forward(self, x):
        self.last_input = x
        self.sum = np.dot(x, self.weights.T) + self.biases
        self.output = self.activation_func(self.sum)
        return self.output

    def backward(self, lr, delta_next):
        local_grad = self.derivative_activation(self.sum)
        delta = local_grad * delta_next

        grad_w = np.dot(delta.T, self.last_input) / self.last_input.shape[0]
        grad_b = np.mean(delta, axis=0)

        self.weights -= lr * grad_w
        self.biases -= lr * grad_b

        delta_prev = np.dot(delta, self.weights)
        return delta_prev

    def __str__(self):
        return f"Layer Name: {self.layer_name}"
