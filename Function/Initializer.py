from enums import InitializerType
import torch

class Initializer:
    """
    Initializer Class.

    This class is used to initialize weights and biases.
    """

    @staticmethod
    def init_weights(weight_init : InitializerType, param_1, param_2, input_size, num_neurons):
        """
        Initialize weights
        :param num_neurons: Number of neutrons
        :param input_size: Size of input
        :param weight_init: Type of weight initialization
        :param param_1: Lower bound or mean
        :param param_2: Upper bound or standard deviation
        :return: Initialized weights
        """
        if weight_init == InitializerType.ZERO:
            return torch.zeros(num_neurons, input_size)
        elif weight_init == InitializerType.RANDOM_DIST_UNIFORM:
            return (torch.rand(num_neurons, input_size) * (param_2 - param_1) + param_1) * torch.sqrt(torch.tensor(2.0 / input_size))
        elif weight_init == InitializerType.RANDOM_DIST_NORMAL:
            return (torch.randn(num_neurons, input_size) * param_2 + param_1) * torch.sqrt(torch.tensor(2.0 / input_size))

    @staticmethod
    def init_bias(bias_init: InitializerType, param_1, param_2, num_neurons):
        """
        Initialize Bias
        :param num_neurons:
        :param bias_init: Type of bias initialization
        :param param_1: Lower bound or mean
        :param param_2: Upper bound or standard deviation
        :return: Initialized bias
        """
        if bias_init == InitializerType.ZERO:
            return torch.zeros(num_neurons)
        elif bias_init == InitializerType.RANDOM_DIST_UNIFORM:
            return (torch.rand(1) * (param_2 - param_1) + param_1) * torch.ones(num_neurons)
        elif bias_init == InitializerType.RANDOM_DIST_NORMAL:
            return (torch.randn(1) * param_2 + param_1) * torch.ones(num_neurons)