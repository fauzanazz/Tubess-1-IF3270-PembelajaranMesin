import numpy as np
import time

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

        for x, y in data_loader():
            output = self.forward(x)
            pred = np.argmax(output, axis=1)

            total += y.shape[0]
            correct += np.sum(pred == y)

        test_time = time.time() - start_time
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%, Time: {test_time:.2f}s")
