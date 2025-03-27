import numpy as np
import time
import matplotlib.pyplot as plt
from Layer import OutputLayer
import pickle
from ANNVisualizer import ANNVisualizer
import os

class ArtificialNeuralNetwork:
    def __init__(self, seeds=0, *layers):
        np.random.seed(seeds)
        self.layers = layers
        self.visualizer = ANNVisualizer(self)

    def batch_generator(self, X, y, batch_size, shuffle=True):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            yield X[batch_indices], y[batch_indices]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, lr, target):
        delta = self.layers[-1].backward(lr, target)
        for layer in reversed(self.layers[:-1]):
            delta = layer.backward(lr, delta)

    def train(self, x, y, loss_function, lr, epochs, verbose=-1, batch_size=32, shuffle=True):
        if isinstance(self.layers[-1], OutputLayer):
            self.layers[-1].loss_funct = loss_function

        training_time = 0
        epoch_times = []
        epoch_losses = []

        train_loader = lambda: self.batch_generator(x, y, batch_size, shuffle)

        for epoch in range(epochs):
            start_time = time.time()
            total_loss = 0.0
            count = 0

            data_batches = train_loader()
            for x_batch, y_batch in data_batches:
                y_onehot = np.zeros((x_batch.shape[0], self.layers[-1].num_neurons))
                y_onehot[np.arange(x_batch.shape[0]), y_batch] = 1
                output = self.forward(x_batch)
                loss = loss_function(y_onehot, output)
                total_loss += loss
                count += 1
                self.backward(lr, y_onehot)

            epoch_time = time.time() - start_time
            training_time += epoch_time
            epoch_times.append(epoch_time)
            avg_loss = total_loss / count if count > 0 else 0
            epoch_losses.append(avg_loss)

            if verbose > 0 and (epoch + 1) % verbose == 0:
                print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Time = {epoch_time:.2f}s")

        print(f"Total training time: {training_time:.2f}s")
        return

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)

    def evaluate(self, x, y):
        y_pred = self.predict(x)
        accuracy = np.mean(y_pred == y)
        return accuracy

    def save(self, filename):
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        filepath = os.path.join(model_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self.layers, f)
        print(f"Model saved to {filepath}")


    def load(self,filename):
        filename = "models/" + filename
        with open(filename, 'rb') as f:
            self.layers = pickle.load(f)
            for layer in self.layers:
                print(layer)
                print(layer.alpha)
        print(f"Model loaded from {filename}")

    def visualize_structure(self):
        return self.visualizer.plot_network_structure()
    
    def visualize_weight_distribution(self, layer_indices):
        self.visualizer.plot_weight_distribution(layer_indices)
    
    def visualize_gradient_distribution(self, layer_indices):
        self.visualizer.plot_gradient_distribution(layer_indices)