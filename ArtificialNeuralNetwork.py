from Layer import Layer, InputLayer, OutputLayer
import torch
from Function import ActivationFunction, LossFunction
from Utils import Utils

class ArtificialNeuralNetwork:
    """
    ArtificialNeuralNetwork Class.

    Consist of many layers of Type Layer.
    """
    def __init__(self, input_size, output_size, hidden_layers, hidden_size, weight_init, bias_init, param_1, param_2, seeds = 0, activation = ActivationFunction.linear, loss_func = LossFunction.mean_squared_error):
        self.layers = []
        self.layers.append(InputLayer(input_size, "Input Layer"))
        for i in range(hidden_layers):
            if i == 0:
                self.layers.append(Layer(weight_init, bias_init, input_size, hidden_size, param_1, param_2, activation, f"Hidden Layer {i}"))
            else:
                self.layers.append(Layer(weight_init, bias_init, hidden_size, hidden_size, param_1, param_2, activation, f"Hidden Layer {i}"))
        self.layers.append(OutputLayer(weight_init, bias_init, hidden_size, output_size, param_1, param_2, activation, "Output Layer"))
        self.loss_func = loss_func
        torch.manual_seed(seeds)
        self.target = None
        self.target_min_output = None

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, lr, target):
        index = len(self.layers) - 1
        for layer in reversed(self.layers):
            if isinstance(layer, InputLayer):
                for layer_up in self.layers:
                    if isinstance(layer_up, InputLayer):
                        continue
                    layer_up.update_bias()
                    layer_up.update_weight()
            else:
                if isinstance(layer, OutputLayer):
                    layer.backward(lr, target)
                else:
                    layer.backward(lr, self.layers[index])
            index -= 1


    def train(self, x, y, lr, epochs, loss_func, verbose = False):
        for epoch in range(epochs):
            loss = 0
            iter = 1
            for x_input, y_input in zip(x, y):
                y_pred = self.forward(x_input)
                loss = loss_func(y_pred, y_input)
                self.target_min_output = Utils.output_minus_target(y_pred, y_input)
                self.backward(lr, y_input)
                if iter % 1000 == 0 and verbose:
                    print(f"Iter {iter} - Loss: {loss}")
                iter += 1
            if verbose:
                print(f"Epoch {epoch} - Loss: {loss}")

    def test(self, x, y, loss_func):
        loss = 0
        for x_input, y_input in zip(x, y):
            y_pred = self.forward(x_input)
            loss += loss_func(y_pred, y_input)
        print(f"Loss: {loss}")