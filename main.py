import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Disable autograd since we are doing manual backpropagation.
torch.set_grad_enabled(False)


# ---------------- Layer Class ----------------
class Layer:
    def __init__(self, num_neurons, input_size, activation, activation_deriv):
        self.weights = torch.randn(num_neurons, input_size) * torch.sqrt(torch.tensor(2.0 / input_size))
        self.biases = torch.zeros(num_neurons)
        self.activation = activation
        self.activation_deriv = activation_deriv
        self.last_input = None
        self.pre_activation = None
        self.output = None

    def forward(self, x):
        self.last_input = x
        self.pre_activation = torch.matmul(x, self.weights.T) + self.biases
        self.output = self.activation(self.pre_activation)
        return self.output

    def backward(self, lr, delta_next):
        local_grad = self.activation_deriv(self.pre_activation)
        delta = local_grad * delta_next

        grad_w = torch.matmul(delta.T, self.last_input) / self.last_input.size(0)
        grad_b = delta.mean(dim=0)

        # Prevent NaN values by clamping gradients
        grad_w = torch.clamp(grad_w, min=-1e10, max=1e10)
        grad_b = torch.clamp(grad_b, min=-1e10, max=1e10)

        self.weights -= lr * grad_w
        self.biases -= lr * grad_b

        delta_prev = torch.matmul(delta, self.weights)
        return delta_prev


# ---------------- OutputLayer Class ----------------
class OutputLayer(Layer):
    def __init__(self, num_neurons, input_size):
        super().__init__(num_neurons, input_size,
                         activation=lambda z: torch.softmax(z, dim=1),
                         activation_deriv=None)

    def backward(self, lr, target):
        delta = self.output - target

        grad_w = torch.matmul(delta.T, self.last_input) / self.last_input.size(0)
        grad_b = delta.mean(dim=0)

        self.weights -= lr * grad_w
        self.biases -= lr * grad_b

        delta_prev = torch.matmul(delta, self.weights)
        return delta_prev


# ---------------- Artificial Neural Network Class ----------------
class ArtificialNeuralNetwork:
    def __init__(self, layers):
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

    def train(self, data_loader, lr, epochs):
        for epoch in range(epochs):
            total_loss = 0.0
            count = 0
            for x, y in data_loader:
                x = x.view(x.size(0), -1)
                y_onehot = torch.zeros(x.size(0), 10)
                y_onehot.scatter_(1, y.view(-1, 1), 1)
                output = self.forward(x)
                loss = -torch.mean(torch.sum(y_onehot * torch.log(output + 1e-7), dim=1))
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
            # Prediction: index of the highest probability.
            pred = torch.argmax(output, dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()
        print(f"Test Accuracy: {100 * correct / total:.2f}%")


# ---------------- Main: Load MNIST and run training/testing ----------------
if __name__ == "__main__":
    # Define a simple transformation.
    transform = transforms.Compose([transforms.ToTensor()])

    # Load MNIST training and test datasets.
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Create data loaders.
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Build the network:
    # - A hidden layer with 128 neurons using ReLU activation.
    # - An output layer with 10 neurons using Softmax activation.
    hidden_layer = Layer(num_neurons=128, input_size=784,
                         activation=torch.relu,
                         activation_deriv=lambda z: (z > 0).float())
    hidden_layer_2 = Layer(num_neurons=128, input_size=128,
                           activation=torch.relu,
                           activation_deriv=lambda z: (
                                   z > 0).float())
    output_layer = OutputLayer(num_neurons=10, input_size=128)

    layers = [hidden_layer, hidden_layer_2, output_layer]

    network = ArtificialNeuralNetwork(layers=layers)

    print("Training network on MNIST (ReLU for hidden and Softmax for output)...")
    network.train(train_loader, lr=0.01, epochs=50)

    print("\nEvaluating on MNIST test set...")
    network.test(test_loader)
