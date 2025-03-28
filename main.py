import sys
sys.dont_write_bytecode = True

# Import libraries
import numpy as np
from Layer.InputLayer import InputLayer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Import custom modules
from ArtificialNeuralNetwork import ArtificialNeuralNetwork
from Layer import Layer, OutputLayer
from Function import ActivationFunction, LossFunction
from enums import InitializerType, RegularizationType


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
    hidden_layers = 5
    hidden_size = 128
    output_size = 10
    learning_rate = 0.1
    param_1 = 0
    param_2 = 0.5
    batch_size = 64

    # Load MNIST dataset using fetch_openml
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X / 255.0
    y = y.astype(int)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    ann = ArtificialNeuralNetwork(
        123,
        InputLayer(input_size=input_size),
        Layer(
            weight_init=InitializerType.XAVIER,
            bias_init=InitializerType.ZERO,
            input_size=input_size,
            num_neurons=20,
            param_1=param_1,
            param_2=param_2,
            activation=ActivationFunction.prelu,
            alpha=0.45,
            layer_name=f"Hidden Layer 0",
        ),
        *[Layer(
            weight_init=InitializerType.XAVIER,
            bias_init=InitializerType.ZERO,
            input_size=20,
            num_neurons=20,
            param_1=param_1,
            param_2=param_2,
            activation=ActivationFunction.prelu,
            alpha=0.45,
            layer_name=f"Hidden Layer {_}",
        ) for _ in range(hidden_layers - 1)],
        OutputLayer(
            weight_init=InitializerType.XAVIER,
            bias_init=InitializerType.ZERO,
            input_size=20,
            num_neurons=output_size,
            param_1=param_1,
            param_2=param_2,
            activation=ActivationFunction.softmax,
            loss_funct=LossFunction.categorical_cross_entropy,
            layer_name="Output Layer"
        )
    )

    ann.train(
        x=X_train,
        y=y_train,
        loss_function=LossFunction.categorical_cross_entropy,
        lr=learning_rate,
        epochs=50,
        batch_size=batch_size,
        verbose=True,
        validation_data=(X_test, y_test),
    )

    print(ann.evaluate(X_test, y_test))

    ann.save("model.pkl")
