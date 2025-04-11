from Function import Initializer
from Function.Regularizer import Regularizer
from enums import RegularizationType
from Function.ActivationFunction import ActivationFunction
import numpy as np

class Layer:
    def __init__(self, weight_init, bias_init, param_1, param_2, num_neurons, input_size,
                 activation, alpha=None, layer_name=None,
                epsilon=1e-8,
                 regularizer: RegularizationType = None,
                 gamma=None,):
        if layer_name is None:
            layer_name = f"layer_{id(self)}"
        self.layer_name = layer_name
        self.activation_func = activation
        self.derivative_activation = None
        self.sum = None
        self.output = None
        self.alpha = None
        self.num_neurons = num_neurons
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
        self.grad_weights = np.zeros_like(self.weights)
        self.last_input = None

        # RMS
        self.epsilon = epsilon
        self.gamma = gamma

        # Regularization
        self.regularizer = Regularizer(regularizer) if regularizer is not None else Regularizer(RegularizationType.NONE)

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

    def _safe_activation(self, x, activation_func, alpha=None):
        try:
            if alpha is not None:
                output = activation_func(x, alpha)
            else:
                output = activation_func(x)

            output = np.nan_to_num(output, nan=0.0, posinf=np.finfo(float).max,neginf=np.finfo(float).min)
            output = np.clip(output,-1e6,1e6)
            return output
        except Exception as e:
            print(f"Error in activation: {e}")
            return np.zeros_like(x)

    def _gradient_clipping(self, gradients, max_norm=1.0):
        grad_norm = np.linalg.norm(gradients)
        if grad_norm > max_norm:
            gradients = gradients * (max_norm / grad_norm)
        return gradients

    def forward(self, x, useRMSprop):
        self.last_input = x
        self.sum = np.nan_to_num(
            np.dot(x, self.weights.T) + self.biases,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6
        )
        if useRMSprop:
            rms = np.sqrt(np.mean(self.sum ** 2, axis=-1, keepdims=True) + self.epsilon)
            normed = self.sum / (rms + self.epsilon)

            if self.gamma is not None:
                normed *= self.gamma

            if self.activation_func in (ActivationFunction.leaky_relu, ActivationFunction.prelu):
                self.output = self._safe_activation(normed, self.activation_func, self.alpha)
            else:
                self.output = self._safe_activation(normed, self.activation_func)
        else:
            if self.activation_func in (ActivationFunction.leaky_relu, ActivationFunction.prelu):
                self.output = self._safe_activation(self.sum, self.activation_func, self.alpha)
            else:
                self.output = self._safe_activation(self.sum, self.activation_func)

        return self.output

    def backward(self, lr, delta_next):
        if self.activation_func in (ActivationFunction.leaky_relu, ActivationFunction.prelu):
            local_grad = self.derivative_activation(self.sum, self.alpha)
        else:
            local_grad = self.derivative_activation(self.sum)

        local_grad = np.nan_to_num(local_grad, nan=0.0, posinf=1.0, neginf=-1.0)
        delta_next = np.nan_to_num(delta_next, nan=0.0, posinf=1.0, neginf=-1.0)

        delta = local_grad * delta_next
        self.grad_weights = np.dot(delta.T, self.last_input) / self.last_input.shape[0]
        grad_b = np.mean(delta, axis=0)

        reg_gradient = self.regularizer.compute_grad(self.weights)
        self.grad_weights += reg_gradient

        delta_prev = np.dot(delta, self.weights)

        self.weights -= lr * self.grad_weights
        self.biases -= lr * grad_b

        return delta_prev
