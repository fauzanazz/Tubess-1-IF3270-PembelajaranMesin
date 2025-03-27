import sys
sys.dont_write_bytecode = True

# Import libraries
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Import custom modules
from ArtificialNeuralNetwork import ArtificialNeuralNetwork
from Layer import Layer, OutputLayer
from Function import ActivationFunction, LossFunction
from enums import InitializerType

def batch_generator(X, y, batch_size, shuffle=True):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]

if __name__ == "__main__":
    input_size = 784
    hidden_layers = 3
    hidden_size = 128
    output_size = 10
    learning_rate = 0.01
    param_1 = 0.01
    param_2 = 0.001
    batch_size = 128

    # Load MNIST dataset using fetch_openml
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X / 255.0
    y = y.astype(int)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # Create train_loader and test_loader as generator objects
    train_loader = lambda: batch_generator(X_train, y_train, batch_size, shuffle=True)
    test_loader = lambda: batch_generator(X_test, y_test, batch_size, shuffle=False)

    ann = ArtificialNeuralNetwork(
        123,
        Layer(
            weight_init=InitializerType.XAVIER,
            bias_init=InitializerType.ZERO,
            input_size=input_size,
            num_neurons=128,
            param_1=param_1,
            param_2=param_2,
            activation=ActivationFunction.relu,
            layer_name=f"Hidden Layer 0"
        ),
        *[Layer(
            weight_init=InitializerType.XAVIER,
            bias_init=InitializerType.ZERO,
            input_size=128,
            num_neurons=128,
            param_1=param_1,
            param_2=param_2,
            activation=ActivationFunction.relu,
            layer_name=f"Hidden Layer 0"
        ) for _ in range(hidden_layers)],
        OutputLayer(
            weight_init=InitializerType.XAVIER,
            bias_init=InitializerType.ZERO,
            input_size=128,
            num_neurons=output_size,
            param_1=param_1,
            param_2=param_2,
            activation=ActivationFunction.softmax,
            loss_funct=LossFunction.categorical_cross_entropy,
            layer_name="Output Layer"
        )
    )

    ann.train(
        train_loader,
        loss_function=LossFunction.categorical_cross_entropy,
        lr=learning_rate,
        epochs=20,
        verbose=1
    )

    ann.test(test_loader)

    ann.save("ann_model.pkl")

    new_model = ArtificialNeuralNetwork()

    new_model.load("ann_model.pkl")

    # new_model.train(
    #     train_loader,
    #     loss_function=LossFunction.categorical_cross_entropy,
    #     lr=0.005,
    #     epochs=50,
    #     verbose=1
    # )
    #
    # new_model.test(test_loader)
    # new_model.visualize_structure()
    new_model.visualize_weight_distribution([0, 1, 2])
    new_model.visualize_gradient_distribution([0, 1, 2])


