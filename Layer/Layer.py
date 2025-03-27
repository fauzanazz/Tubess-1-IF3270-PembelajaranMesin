from Function import Initializer
from enums import InitializerType
from Function.ActivationFunction import ActivationFunction
import numpy as np

class Layer:
    def __init__(self, weight_init: InitializerType, bias_init: InitializerType, 
                param_1, param_2, num_neurons, input_size, 
                activation=ActivationFunction.linear, alpha = None, layer_name=None):
        self.layer_name = layer_name
        self.activation_func = activation
        self.derivative_activation = None
        self.sum = None
        self.output = None
        self.alpha = None

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

        self.last_input = None

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
        elif activation == ActivationFunction.leaky_relu:
            self.derivative_activation = ActivationFunction.derivative_leaky_relu
            self.alpha = 0.01 if alpha is None else alpha
        elif activation == ActivationFunction.prelu:
            self.derivative_activation = ActivationFunction.derivative_prelu
            self.alpha = 0.01 if alpha is None else alpha
        else:
            raise ValueError(f"Activation function {activation} is not supported")

    def forward(self, x):
        self.last_input = x
        self.sum = np.dot(x, self.weights.T) + self.biases
        if self.alpha is not None:
            self.output = ActivationFunction.prelu(self.sum, self.alpha)
        else:
            self.output = self.activation_func(self.sum)
        return self.output

    def backward(self, lr, delta_next):
        if self.alpha is not None:
            local_grad = self.derivative_activation(self.sum, self.alpha)
        else:
            local_grad = self.derivative_activation(self.sum)

        local_grad = np.nan_to_num(local_grad, nan=0.0, posinf=1.0, neginf=-1.0)
        delta_next = np.nan_to_num(delta_next, nan=0.0, posinf=1.0, neginf=-1.0)

        delta = local_grad * delta_next

        grad_w = np.dot(delta.T, self.last_input) / self.last_input.shape[0]
        grad_b = np.mean(delta, axis=0)

        self.weights -= lr * grad_w
        self.biases -= lr * grad_b

        delta_prev = np.dot(delta, self.weights)
        return delta_prev

    def __str__(self):
        return f"Layer Name: {self.layer_name}"
