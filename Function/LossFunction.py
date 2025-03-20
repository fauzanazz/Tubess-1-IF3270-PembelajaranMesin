from numba import njit
import numpy as np


class LossFunction:
    @staticmethod
    @njit(cache=True, fastmath=True)
    def mean_squared_error(y_pred, y_true, epsilon=1e-7):
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        y_true = np.clip(y_true, epsilon, 1.0 - epsilon)
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    @njit(cache=True, fastmath=True)
    def mean_squared_error_derivative(y_pred, y_true, epsilon=1e-7):
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        y_true = np.clip(y_true, epsilon, 1.0 - epsilon)
        return 2 * (y_pred - y_true) / y_true.shape[0]

    @staticmethod
    @njit(cache=True, fastmath=True)
    def binary_cross_entropy(y_pred, y_true, epsilon=1e-7):
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        y_true = np.clip(y_true, epsilon, 1.0 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    @njit(cache=True, fastmath=True)
    def binary_cross_entropy_derivative(y_pred, y_true, epsilon=1e-7):
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        y_true = np.clip(y_true, epsilon, 1.0 - epsilon)
        return (y_pred - y_true) / y_true.shape[0]

    @staticmethod
    @njit(cache=True, fastmath=True)
    def categorical_cross_entropy(y_pred, y_true, epsilon=1e-7):
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)

        # Handle different dimensions
        if y_pred.ndim == 1:
            # For 1D arrays (binary case)
            y_true = np.clip(y_true, epsilon, 1.0 - epsilon)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            # For 2D arrays (multi-class case)
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    @staticmethod
    @njit(cache=True, fastmath=True)
    def categorical_cross_entropy_derivative(y_pred, y_true, epsilon=1e-7):
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        gradient = y_pred - y_true
        return gradient / y_true.shape[0]

    