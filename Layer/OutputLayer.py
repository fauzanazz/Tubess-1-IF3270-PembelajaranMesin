from Function import LossFunction
from Layer import Layer
from enums import InitializerType, RegularizationType
from Function.ActivationFunction import ActivationFunction
import numpy as np

class OutputLayer(Layer):
    def __init__(self, weight_init: InitializerType, bias_init: InitializerType, 
                input_size, num_neurons, param_1, param_2, 
                activation=ActivationFunction.linear, layer_name=None, loss_funct = None,
                decay_rate=0.9, epsilon=1e-8,
                regularizer: RegularizationType = None,
                 optimizer="sgd", lr=0.001, beta1=0.9, beta2=0.999
                 ):
        super().__init__(
            weight_init=weight_init,
            bias_init=bias_init,
            input_size=input_size,
            num_neurons=num_neurons,
            param_1=param_1,
            param_2=param_2,
            activation=activation,
            layer_name=layer_name,
            decay_rate=decay_rate,
            regularizer=regularizer,
            optimizer=optimizer,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
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

        delta = np.nan_to_num(delta, nan=0.0, posinf=1.0, neginf=-1.0)

        self.grad_weights = np.dot(delta.T, self.last_input) / self.last_input.shape[0]
        grad_b = np.mean(delta, axis=0)

        reg_gradient = self.regularizer.compute_grad(self.weights)
        self.grad_weights += reg_gradient

        delta_prev = np.dot(delta, self.weights)
        self.weights, self.biases = self.optimizer.update(
            layer_name=self.layer_name,
            weights=self.weights,
            biases=self.biases,
            grad_weights=self.grad_weights,
            grad_biases=grad_b
        )

        return delta_prev

