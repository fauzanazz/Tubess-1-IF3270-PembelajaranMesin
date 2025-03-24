import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class ANNVisualizer:
    def __init__(self, model):
        self.model = model
    
    def plot_network_structure(self):
        """Menampilkan struktur jaringan dengan bobot dan gradien dalam bentuk graf."""
        G = nx.DiGraph()
        pos = {}
        layer_sizes = []
        
        # Menentukan jumlah neuron di tiap layer
        for i, layer in enumerate(self.model.layers):
            num_neurons = layer.weights.shape[0]
            layer_sizes.append(num_neurons)
            for j in range(num_neurons):
                node_label = f'L{i}N{j}'
                G.add_node(node_label, layer=i)
                pos[node_label] = (i, -j)
                
                if i > 0:
                    for k in range(layer.weights.shape[1]):
                        prev_label = f'L{i-1}N{k}'
                        weight = layer.weights[j, k]
                        G.add_edge(prev_label, node_label, weight=weight)
        
        # Plot
        plt.figure(figsize=(10, 6))
        edges = G.edges(data=True)
        weights = [d['weight'] for _, _, d in edges]
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color=weights, edge_cmap=plt.cm.Blues)
        plt.title("ANN Structure with Weights")
        plt.show()
    
    def plot_weight_distribution(self, layer_indices):
        """Menampilkan distribusi bobot dari layer tertentu."""
        for idx in layer_indices:
            if idx >= len(self.model.layers):
                print(f"Layer {idx} tidak ditemukan.")
                continue
            
            weights = self.model.layers[idx].weights.flatten()
            plt.figure(figsize=(6, 4))
            plt.hist(weights, bins=30, alpha=0.7, color='blue', edgecolor='black')
            plt.title(f'Distribusi Bobot - Layer {idx}')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.show()
    
    def plot_gradient_distribution(self, layer_indices):
        """Menampilkan distribusi gradien bobot dari layer tertentu."""
        for idx in layer_indices:
            if idx >= len(self.model.layers):
                print(f"Layer {idx} tidak ditemukan.")
                continue
            
            gradients = self.model.layers[idx].weights.flatten()
            plt.figure(figsize=(6, 4))
            plt.hist(gradients, bins=30, alpha=0.7, color='red', edgecolor='black')
            plt.title(f'Distribusi Gradien Bobot - Layer {idx}')
            plt.xlabel('Gradient Value')
            plt.ylabel('Frequency')
            plt.show()