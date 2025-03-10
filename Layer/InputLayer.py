import torch

class InputLayer:
    """
    InputLayer Class.

    Consist of many neurons of Input Layer
    This layer only store input without weight.
    """

    def __init__(self, input_size, layer_name = None):
        self.input_size = input_size
        self.input = torch.zeros(input_size)
        self.layer_name = layer_name

    def forward(self, x):
        self.input = x
        return x

    def __str__(self):
        return f"Layer Name: {self.layer_name}\nNeurons: {self.input_size}\n"