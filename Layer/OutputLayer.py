from Function import LossFunction
from Layer import Layer
from enums import InitializerType
from Function.ActivationFunction import ActivationFunction
import numpy as np

class OutputLayer(Layer):
    def __init__(self, weight_init: InitializerType, bias_init: InitializerType, 
                input_size, num_neurons, param_1, param_2, 
                activation=ActivationFunction.linear, layer_name=None, loss_funct = None):
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

        match loss_funct:
            case LossFunction.mean_squared_error:
                self.derivative_loss = LossFunction.mean_squared_error_derivative
            case LossFunction.binary_cross_entropy:
                self.derivative_loss = LossFunction.binary_cross_entropy_derivative
            case LossFunction.categorical_cross_entropy:
                self.derivative_loss = LossFunction.categorical_cross_entropy_derivative
            case _:
                raise NotImplementedError

    def backward(self, lr, target):
        delta = self.derivative_loss(self.output, target)

        grad_w = np.dot(delta.T, self.last_input) / self.last_input.shape[0]
        grad_b = np.mean(delta, axis=0)

        self.weights -= lr * grad_w
        self.biases -= lr * grad_b

        delta_prev = np.dot(delta, self.weights)
        return delta_prev

