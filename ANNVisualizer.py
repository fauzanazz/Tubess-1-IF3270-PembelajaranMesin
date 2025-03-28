import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class ANNVisualizer:
    def __init__(self, model):
        self.model = model

    def plot_network_structure(self):
        """Visualisasi ANN structure dengan bobot menggunakan NetworkX."""
        G = nx.DiGraph()
        pos = {}
        
        # Loop over layers
        for i, layer in enumerate(self.model.layers):
            # Determine the number of neurons.
            if not hasattr(layer, 'weights') or layer.weights is None:
                num_neurons = layer.num_neurons
            else:
                # For layers with weights, assume weight matrix shape is (num_neurons, num_neurons_prev)
                num_neurons = layer.weights.shape[0]
                
            for j in range(num_neurons):
                # Use a label that indicates layer and neuron index.
                node_label = f"L{i}N{j}"
                G.add_node(node_label, layer=i)
                pos[node_label] = (i, -j)
                
                # For layers beyond the first, add edges from the previous layer.
                if i > 0 and hasattr(layer, 'weights') and layer.weights is not None:
                    num_prev = self.model.layers[i-1].num_neurons
                    for k in range(num_prev):
                        prev_label = f"L{i-1}N{k}"
                        if layer.weights.shape[1] > k and layer.weights.shape[0] > j:
                            weight = layer.weights[j, k]
                        else:
                            weight = 0
                        G.add_edge(prev_label, node_label, weight=weight)
        
        # Plot the network graph.
        plt.figure(figsize=(10, 6))
        edges = G.edges(data=True)
        edge_weights = [d['weight'] for _, _, d in edges]
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                edge_color=edge_weights, edge_cmap=plt.cm.Blues, width=1.5)
        plt.title("ANN Structure with Weights")
        plt.show()
    
    def plot_layer_tables(self):
        """Plot tabel average bobot dan gradient untuk setiap layer"""
        for i, layer in enumerate(self.model.layers):
            if i == 0:
                continue  # Skip input layer

            # Determine how many neurons this layer has
            if not hasattr(layer, 'weights') or layer.weights is None:
                num_neurons = layer.num_neurons
            else:
                num_neurons = layer.weights.shape[0]

            # Build table data
            table_data = []
            for j in range(num_neurons):
                neuron_label = f"N{j}"
                # Average weight
                if hasattr(layer, 'weights') and layer.weights is not None:
                    avg_weight = np.mean(layer.weights[j, :])
                else:
                    avg_weight = "N/A"
                # Average grad weight
                if hasattr(layer, 'grad_weights') and layer.grad_weights is not None:
                    avg_grad = np.mean(layer.grad_weights[j, :])
                else:
                    avg_grad = "N/A"

                avg_weight_str = f"{avg_weight:.4f}" if isinstance(avg_weight, float) else avg_weight
                avg_grad_str = f"{avg_grad:.4f}" if isinstance(avg_grad, float) else avg_grad
                table_data.append([neuron_label, avg_weight_str, avg_grad_str])

            base_width = 6
            base_height = 2
            row_height_factor = 0.25  # additional height per neuron

            fig_height = base_height + (num_neurons * row_height_factor)
            fig, ax = plt.subplots(figsize=(base_width, fig_height))

            ax.axis('off')
            ax.set_title(f"Layer {i} Weights Summary", fontsize=10, pad=10, y=1.02)

            col_labels = ["Neuron", "Avg Weight", "Avg Grad Weight"]
            tbl = ax.table(
                cellText=table_data,
                colLabels=col_labels,
                loc='center',
                cellLoc='center',
                edges='closed'
            )

            # Styling
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            for (row, col), cell in tbl.get_celld().items():
                cell.set_edgecolor('black')
                cell.set_linewidth(1)

            for col_idx in range(len(col_labels)):
                tbl.auto_set_column_width(col_idx)

            plt.tight_layout()
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
            
            gradients = self.model.layers[idx].grad_weights.flatten()
            plt.figure(figsize=(6, 4))
            plt.hist(gradients, bins=30, alpha=0.7, color='red', edgecolor='black')
            plt.title(f'Distribusi Gradien Bobot - Layer {idx}')
            plt.xlabel('Gradient Value')
            plt.ylabel('Frequency')
            plt.show()