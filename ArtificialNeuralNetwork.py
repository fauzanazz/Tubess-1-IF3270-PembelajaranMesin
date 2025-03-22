import numpy as np
import time
import matplotlib.pyplot as plt
from Layer import OutputLayer
import pickle

class ArtificialNeuralNetwork:
    def __init__(self, seeds=0, *layers):
        np.random.seed(seeds)
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, lr, target):
        delta = self.layers[-1].backward(lr, target)
        for layer in reversed(self.layers[:-1]):
            delta = layer.backward(lr, delta)

    def train(self, data_loader, loss_function, lr, epochs, verbose=-1):
        if isinstance(self.layers[-1], OutputLayer):
            self.layers[-1].loss_funct = loss_function

        training_time = 0
        epoch_times = []
        epoch_losses = []

        for epoch in range(epochs):
            start_time = time.time()
            total_loss = 0.0
            count = 0

            for x, y in data_loader():
                y_onehot = np.zeros((x.shape[0], self.layers[-1].num_neurons))
                y_onehot[np.arange(x.shape[0]), y] = 1

                output = self.forward(x)
                loss = loss_function(y_onehot, output)
                total_loss += loss
                count += 1
                self.backward(lr, y_onehot)

            epoch_time = time.time() - start_time
            training_time += epoch_time
            epoch_times.append(epoch_time)
            avg_loss = total_loss / count
            epoch_losses.append(avg_loss)

            if verbose > 0 and (epoch + 1) % verbose == 0:
                print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Time = {epoch_time:.2f}s")

        print(f"Total training time: {training_time:.2f}s")
        return

    def test(self, data_loader):

        correct = 0
        total = 0
        start_time = time.time()

        # Take one batch for visualization
        visualize_batch = None
        visualize_preds = None
        visualize_labels = None

        for x, y in data_loader():
            output = self.forward(x)
            pred = np.argmax(output, axis=1)

            total += y.shape[0]
            correct += np.sum(pred == y)

            # Save first batch for visualization
            if visualize_batch is None:
                visualize_batch = x[:25].copy()  # Take first 25 examples
                visualize_preds = pred[:25].copy()
                visualize_labels = y[:25].copy()

        test_time = time.time() - start_time
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%, Time: {test_time:.2f}s")

        # Display the images, predictions, and ground truth
        if visualize_batch is not None:
            plt.figure(figsize=(10, 10))
            for i in range(min(25, len(visualize_batch))):
                plt.subplot(5, 5, i+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)

                # Reshape if the input is flattened (e.g., MNIST)
                if visualize_batch[i].shape[0] == 784:
                    plt.imshow(visualize_batch[i].reshape(28, 28), cmap='gray')
                else:
                    plt.imshow(visualize_batch[i], cmap='gray')

                color = 'green' if visualize_preds[i] == visualize_labels[i] else 'red'
                plt.title(f"Pred: {visualize_preds[i]}\nTrue: {visualize_labels[i]}",
                          color=color)

            plt.tight_layout()
            plt.show()

    def save(self, filename):
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        filename = "models/" + filename
        with open(filename, 'wb') as f:
            pickle.dump(self.layers, f)
        print(f"Model saved to {filename}")


    def load(self,filename):
        filename = "models/" + filename
        with open(filename, 'rb') as f:
            self.layers = pickle.load(f)
            for layer in self.layers:
                print(layer)
                print(layer.alpha)
        print(f"Model loaded from {filename}")
