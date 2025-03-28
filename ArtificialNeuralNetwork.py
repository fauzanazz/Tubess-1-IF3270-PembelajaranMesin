import numpy as np
import time
from Layer import OutputLayer
import pickle
from ANNVisualizer import ANNVisualizer
import os
from tqdm.auto import tqdm
from collections import defaultdict


class ArtificialNeuralNetwork:
    def __init__(self, seeds=0, *layers):
        np.random.seed(seeds)
        self.layers = layers
        self.visualizer = ANNVisualizer(self)

    def batch_generator(self, X, y, batch_size, shuffle=True):
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"

        indices_by_class = defaultdict(list)
        for idx, label in enumerate(y):
            indices_by_class[label].append(idx)

        if shuffle:
            for label in indices_by_class:
                np.random.shuffle(indices_by_class[label])

        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        splits_by_class = {}
        for label, indices in indices_by_class.items():
            splits_by_class[label] = np.array_split(indices, n_batches)

        for i in range(n_batches):
            batch_indices = []
            for label in splits_by_class:
                if i < len(splits_by_class[label]):
                    batch_indices.extend(splits_by_class[label][i])
            if len(batch_indices) > batch_size:
                if shuffle:
                    np.random.shuffle(batch_indices)
                batch_indices = batch_indices[:batch_size]
            yield X[batch_indices], y[batch_indices]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, lr, target):
        delta = self.layers[-1].backward(lr, target)
        for layer in reversed(self.layers[:-1]):
            delta = layer.backward(lr, delta)

    def train(self, x, y, loss_function, lr, epochs, verbose=False, batch_size=32, shuffle=True, validation_data=None):
        if isinstance(self.layers[-1], OutputLayer):
            self.layers[-1].loss_funct = loss_function

        training_time = 0
        epoch_times = []
        epoch_losses = []
        val_losses = []

        has_validation = validation_data is not None
        if has_validation:
            x_val, y_val = validation_data

        epochs_iter = tqdm(range(epochs), desc="Training", disable=not verbose)
        for _ in epochs_iter:
            start_time = time.time()
            total_loss = 0.0
            count = 0


            for x_batch, y_batch in self.batch_generator(x, y, batch_size, shuffle):
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

            val_loss = None
            if has_validation:
                y_val_onehot = np.zeros((x_val.shape[0], self.layers[-1].num_neurons))
                y_val_onehot[np.arange(x_val.shape[0]), y_val] = 1
                val_output = self.forward(x_val)
                val_loss = loss_function(y_val_onehot, val_output)
                val_losses.append(val_loss)

            if verbose:
                status = f"Loss: {avg_loss:.4f}"
                if has_validation:
                    status += f", Val Loss: {val_loss:.4f}"
                epochs_iter.set_postfix_str(status)

        print(f"Total training time: {training_time:.2f}s")
        return (epoch_losses, val_losses) if has_validation else epoch_losses

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
        print(f"Model loaded from {filename}")

    def visualize_structure(self):
        return self.visualizer.plot_network_structure()
    
    def visualize_weight_table(self):
        self.visualizer.plot_layer_tables()
    
    def visualize_weight_distribution(self, layer_indices):
        self.visualizer.plot_weight_distribution(layer_indices)
    
    def visualize_gradient_distribution(self, layer_indices):
        self.visualizer.plot_gradient_distribution(layer_indices)