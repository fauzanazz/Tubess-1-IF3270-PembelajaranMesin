import torch

class ArtificialNeuralNetwork:
    """
    ArtificialNeuralNetwork Class.

    Consist of many layers of Type Layer.
    """
    def __init__(self, seeds = 0, *layers):
        torch.manual_seed(seeds)
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, lr, target):
        # Backpropagate starting from the output layer.
        delta = self.layers[-1].backward(lr, target)
        for layer in reversed(self.layers[:-1]):
            delta = layer.backward(lr, delta)


    def train(self, data_loader, loss_function, lr, epochs):
        for epoch in range(epochs):
            total_loss = 0.0
            count = 0
            for x, y in data_loader:
                x = x.view(x.size(0), -1)
                y_onehot = torch.zeros(x.size(0), self.layers[-1].num_neurons)
                y_onehot.scatter_(1, y.view(-1, 1), 1)
                output = self.forward(x)
                loss = loss_function(y_onehot, output)
                total_loss += loss.item()
                count += 1
                self.backward(lr, y_onehot)
            print(f"Epoch {epoch + 1}: Loss = {total_loss / count:.4f}")

    def test(self, data_loader):
        correct = 0
        total = 0
        for x, y in data_loader:
            x = x.view(x.size(0), -1)
            output = self.forward(x)
            pred = torch.argmax(output, dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()
        print(f"Test Accuracy: {100 * correct / total:.2f}%")