import sys
sys.dont_write_bytecode = True

# Import libraries
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


from ArtificialNeuralNetwork import ArtificialNeuralNetwork
from Layer import Layer, OutputLayer
from Function import ActivationFunction, LossFunction
from enums import InitializerType

if __name__ == "__main__":
    # Disable __grad based on assistant direction
    torch.set_grad_enabled(False)

    # Const variables
    input_size = 784
    hidden_layers = 2
    hidden_size = 128
    output_size = 10
    learning_rate = 0.001
    param_1 = 0
    param_2 = 0

    # Import Dataset
    train = pd.read_csv("data/train.csv")

    # Define a simple transformation.
    transform = transforms.Compose([transforms.ToTensor()])

    # Load MNIST training and test datasets.
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Create data loaders.
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize Artificial Neural Network
    ann = ArtificialNeuralNetwork(
        123,
        Layer(
            weight_init=InitializerType.RANDOM_DIST_UNIFORM,
            bias_init=InitializerType.ZERO,
            input_size=input_size,
            num_neurons=hidden_size,
            param_1=param_1,
            param_2=param_2,
            activation=ActivationFunction.relu,
            layer_name=f"Hidden Layer 0"
        ),
        *[Layer(
            weight_init=InitializerType.RANDOM_DIST_UNIFORM,
            bias_init=InitializerType.ZERO,
            input_size=hidden_size,
            num_neurons=hidden_size,
            param_1=param_1,
            param_2=param_2,
            activation=ActivationFunction.relu,
            layer_name=f"Hidden Layer {i + 1}"
        ) for i in range(hidden_layers)],
        OutputLayer(
            weight_init=InitializerType.RANDOM_DIST_UNIFORM,
            bias_init=InitializerType.ZERO,
            input_size=hidden_size,
            num_neurons=output_size,
            param_1=param_1,
            param_2=param_2,
            activation=ActivationFunction.relu,
            layer_name="Output Layer"
        )
    )

    # Train Artificial Neural Network
    ann.train(
        train_loader,
        lr=0.01,
        epochs=10,
        loss_function=LossFunction.categorical_cross_entropy
    )

    ann.test(test_loader)
