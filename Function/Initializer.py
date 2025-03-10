from enums import InitializerType
import torch

class Initializer:
    """
    Initializer Class.

    This class is used to initialize weights and biases.
    """

    @staticmethod
    def init_weights(weight_init : InitializerType, param_1, param_2, size: int):
        """
        Initialize weights
        :param weight_init: Type of weight initialization
        :param param_1: Lower bound or mean
        :param param_2: Upper bound or standard deviation
        :param size: Size of the weight
        :return: Initialized weights
        """
        if weight_init == InitializerType.ZERO:
            return Initializer.zero_init(size)
        elif weight_init == InitializerType.RANDOM_DIST_UNIFORM:
            return Initializer.random_dist_uniform(size, param_1, param_2)
        elif weight_init == InitializerType.RANDOM_DIST_NORMAL:
            return Initializer.random_dist_normal(size, param_1, param_2)

    @staticmethod
    def init_bias(bias_init: InitializerType, param_1, param_2):
        """
        Initialize Bias
        :param bias_init: Type of bias initialization
        :param param_1: Lower bound or mean
        :param param_2: Upper bound or standard deviation
        :return: Initialized bias
        """
        if bias_init == InitializerType.ZERO:
            return 0
        elif bias_init == InitializerType.RANDOM_DIST_UNIFORM:
            return torch.rand(1) * (param_2 - param_1) + param_1
        elif bias_init == InitializerType.RANDOM_DIST_NORMAL:
            return torch.randn(1) * param_2 + param_1


    @staticmethod
    def zero_init(size):
        return torch.zeros(size)

    @staticmethod
    def random_dist_uniform(size, lower_bound, upper_bound):
        return torch.rand(size) * (upper_bound - lower_bound) + lower_bound

    @staticmethod
    def random_dist_normal(size, mean, std):
        return torch.randn(size) * std + mean