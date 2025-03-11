from Layer import InputLayer, OutputLayer
import torch
from Function import LossFunction
from Utils import Utils

class ArtificialNeuralNetwork:
    """
    ArtificialNeuralNetwork Class.

    Consist of many layers of Type Layer.
    """
    def __init__(self, seeds = 0, loss_func = LossFunction.mean_squared_error, *layers):
        self.layers = layers
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
                    layer.backward(lr, self.layers[index+1])
            index -= 1


    def train(self, x, y, lr, epochs, loss_func, verbose = False):
        iter = 1
        for epoch in range(epochs):
            loss = 0
            for x_input, y_input in zip(x, y):
                y_pred = self.forward(x_input)
                loss += loss_func(y_pred, y_input)
                self.target_min_output = Utils.output_minus_target(y_pred, y_input)
                self.backward(lr, y_input)
                iter += 1
            if verbose:
                print(f"Epoch {epoch} - Loss: {loss/iter}")

    def test(self, x, y, loss_func):
        loss = 0
        for x_input, y_input in zip(x, y):
            y_pred = self.forward(x_input)
            print(f"Prediction: {torch.argmax(y_pred)} - Target: {y_input}")
            loss += loss_func(y_pred, y_input)
        print(f"Loss: {loss/len(x)}")