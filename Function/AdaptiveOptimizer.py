import numpy as np
from enum import Enum


class OptimizerType(Enum):
    SGD = "sgd"
    ADAM = "adam"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"


class AdaptiveOptimizer:
    def __init__(self, method=OptimizerType.SGD, lr=0.001,
                 beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.method = method if isinstance(method, OptimizerType) else OptimizerType(method.lower())
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.moments = {}
        self.iterations = {}

    def initialize_layer(self, layer_name, weight_shape):
        """Initialize optimizer state for a layer"""
        if layer_name in self.moments:
            return

        self.moments[layer_name] = {
            'weights': {},
            'biases': {}
        }

        if self.method == OptimizerType.ADAM:
            self.moments[layer_name]['weights']['m'] = np.zeros(weight_shape)
            self.moments[layer_name]['weights']['v'] = np.zeros(weight_shape)
            self.moments[layer_name]['biases']['m'] = np.zeros(weight_shape[0])
            self.moments[layer_name]['biases']['v'] = np.zeros(weight_shape[0])
            self.iterations[layer_name] = 0

        elif self.method == OptimizerType.RMSPROP:
            self.moments[layer_name]['weights']['rms'] = np.zeros(weight_shape)
            self.moments[layer_name]['biases']['rms'] = np.zeros(weight_shape[0])

        elif self.method == OptimizerType.ADAGRAD:
            self.moments[layer_name]['weights']['cache'] = np.zeros(weight_shape)
            self.moments[layer_name]['biases']['cache'] = np.zeros(weight_shape[0])

    def update(self, layer_name, weights, biases, grad_weights, grad_biases):
        self.initialize_layer(layer_name, grad_weights.shape)

        if self.method == OptimizerType.ADAM:
            return self._update_adam(layer_name, weights, biases, grad_weights, grad_biases)
        elif self.method == OptimizerType.RMSPROP:
            return self._update_rmsprop(layer_name, weights, biases, grad_weights, grad_biases)
        elif self.method == OptimizerType.ADAGRAD:
            return self._update_adagrad(layer_name, weights, biases, grad_weights, grad_biases)
        else:
            return weights - self.lr * grad_weights, biases - self.lr * grad_biases

    def _update_adam(self, layer_name, weights, biases, grad_weights, grad_biases):
        self.iterations[layer_name] += 1
        t = self.iterations[layer_name]

        self.moments[layer_name]['weights']['m'] = (
                self.beta1 * self.moments[layer_name]['weights']['m'] +
                (1 - self.beta1) * grad_weights
        )
        self.moments[layer_name]['biases']['m'] = (
                self.beta1 * self.moments[layer_name]['biases']['m'] +
                (1 - self.beta1) * grad_biases
        )

        self.moments[layer_name]['weights']['v'] = (
                self.beta2 * self.moments[layer_name]['weights']['v'] +
                (1 - self.beta2) * (grad_weights ** 2)
        )
        self.moments[layer_name]['biases']['v'] = (
                self.beta2 * self.moments[layer_name]['biases']['v'] +
                (1 - self.beta2) * (grad_biases ** 2)
        )

        m_hat_w = self.moments[layer_name]['weights']['m'] / (1 - self.beta1 ** t)
        m_hat_b = self.moments[layer_name]['biases']['m'] / (1 - self.beta1 ** t)
        v_hat_w = self.moments[layer_name]['weights']['v'] / (1 - self.beta2 ** t)
        v_hat_b = self.moments[layer_name]['biases']['v'] / (1 - self.beta2 ** t)

        weights_update = self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
        biases_update = self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

        return weights - weights_update, biases - biases_update

    def _update_rmsprop(self, layer_name, weights, biases, grad_weights, grad_biases):
        self.moments[layer_name]['weights']['rms'] = (
                0.9 * self.moments[layer_name]['weights']['rms'] +
                0.1 * (grad_weights ** 2)
        )
        self.moments[layer_name]['biases']['rms'] = (
                0.9 * self.moments[layer_name]['biases']['rms'] +
                0.1 * (grad_biases ** 2)
        )

        weights_update = (
                self.lr * grad_weights /
                (np.sqrt(self.moments[layer_name]['weights']['rms']) + self.epsilon)
        )
        biases_update = (
                self.lr * grad_biases /
                (np.sqrt(self.moments[layer_name]['biases']['rms']) + self.epsilon)
        )

        return weights - weights_update, biases - biases_update

    def _update_adagrad(self, layer_name, weights, biases, grad_weights, grad_biases):
        self.moments[layer_name]['weights']['cache'] += grad_weights ** 2
        self.moments[layer_name]['biases']['cache'] += grad_biases ** 2

        weights_update = (
                self.lr * grad_weights /
                (np.sqrt(self.moments[layer_name]['weights']['cache']) + self.epsilon)
        )
        biases_update = (
                self.lr * grad_biases /
                (np.sqrt(self.moments[layer_name]['biases']['cache']) + self.epsilon)
        )

        return weights - weights_update, biases - biases_update
