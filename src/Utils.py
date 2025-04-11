import torch

class Utils:
    """
    Encoder Class

    Encode values
    """
    @staticmethod
    def output_minus_target(output, target):
        """
        Calculate output minus target
        :param output: Tensor Array of output
        :param target: Int target
        :return: Tensor Array of output minus target
        """
        target_array = torch.zeros_like(output)
        target_array[target] = 1
        return output - target_array