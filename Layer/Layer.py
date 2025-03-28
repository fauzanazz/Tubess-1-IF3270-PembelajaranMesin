from Function import Initializer
from Function.AdaptiveOptimizer import AdaptiveOptimizer
from Function.Regularizer import Regularizer
from enums import RegularizationType
from Function.ActivationFunction import ActivationFunction
import numpy as np

class Layer:
    def __init__(self, weight_init, bias_init, param_1, param_2, num_neurons, input_size,
                 activation, alpha=None, layer_name=None,
                 decay_rate=0.9, epsilon=1e-8,
                 regularizer: RegularizationType = None,
                 optimizer="sgd", lr = 0.001, beta1=0.9, beta2=0.999):
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
        self.weight_rms = np.zeros_like(self.weights)
        self.bias_rms = np.zeros_like(self.biases)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        # Regularization
        self.regularizer = Regularizer(regularizer) if regularizer is not None else Regularizer(RegularizationType.NONE)

        # Optimizer
        self.optimizer = AdaptiveOptimizer(
            method=optimizer,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
        )

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
        if self.activation_func in (ActivationFunction.leaky_relu, ActivationFunction.prelu):
            self.output = self.activation_func(self.sum, self.alpha)
        else:
            self.output = self.activation_func(self.sum)
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

        self.weights, self.biases = self.optimizer.update(
            layer_name=self.layer_name,
            weights=self.weights,
            biases=self.biases,
            grad_weights=self.grad_weights,
            grad_biases=grad_b
        )

        return delta_prev
