"""
A module for plotting graphs and charts.
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import networkx as nx
import numpy as np
import os
from sklearn import preprocessing

"""
Global variables for plotting
"""
default_nodes_cmap = "tab20"

edge_colors = [(0, "blue"), (0.5, "black"), (1, "red")]
"""
Custom colormap for edges based on Ricci curvature values. The colors represent different curvature ranges:
- Blue: Negative curvature
- Black: Zero curvature
- Red: Positive curvature
"""
curvature_cmap = mcolors.LinearSegmentedColormap.from_list("CustomMap", edge_colors)

class GraphDrawer:
    """
    A class for visualizing network graphs with Ricci curvature and community structures.
    """
    def __init__(self, graph, title, save_path, seed=42):
        """
        Initialize the GraphDrawer with a graph, title, save path, and an optional seed for layout positioning.

        :param graph: The graph to be visualized.
        :type graph: networkx.Graph
        :param title: The title for the plot.
        :type title: str
        :param save_path: Directory where the plot will be saved.
        :type save_path: str
        :param seed: Seed for the layout positioning. Default is 42.
        :type seed: int, optional
        """
        self.graph = graph
        self.title = title
        self.save_path = save_path
        self.pos = nx.spring_layout(graph, center=(0, 0), seed=seed)

    def draw_colorbar(self, curvature_values):
        """
        Draws a colorbar for the Ricci curvature values.

        :param curvature_values: List of curvature values for edges.
        :type curvature_values: list of float
        """
        sm = plt.cm.ScalarMappable(
            cmap=curvature_cmap,
            norm=plt.Normalize(vmin=min(curvature_values), vmax=max(curvature_values)),
        )
        plt.colorbar(sm, ax=plt.gca(), label="Ricci Curvature")

    def save_and_show(self, plot_axis=False):
        """
        Saves the current plot to the specified directory and displays it.

        :param plot_axis: Whether to display the axis in the plot. Default is False.
        :type plot_axis: bool, optional
        """
        plt.gcf().canvas.manager.set_window_title(self.title)
        if not plot_axis:
            plt.axis("off")
        plt.savefig(os.path.join(self.save_path, self.title + ".png"), dpi=600)
        plt.show()

    def draw_graph(self, clustering_label, nodes_cmap=default_nodes_cmap):
        """
        Draws the graph with community coloring (from ground truth) and Ricci curvature visualization.

        :param clustering_label: Node attribute name for clustering (community) labels.
        :type clustering_label: str
        :param nodes_cmap: The colormap for nodes. Default is `default_nodes_cmap`.
        :type nodes_cmap: str, optional
        """
        complex_list = nx.get_node_attributes(self.graph, clustering_label)
        le = preprocessing.LabelEncoder()
        node_color = le.fit_transform(list(complex_list.values()))

        colormap = mcm.get_cmap(nodes_cmap)  # Discrete colormap
        # Create node colors based on community assignment
        mapped_node_color = [colormap(comm) for comm in node_color]

        curvature_values = [
            d["ricciCurvature"] for _, _, d in self.graph.edges(data=True)
        ]
        # Add color bar for Ricci Curvature
        self.draw_colorbar(curvature_values)
        # Normalize curvature values between 0 and 1
        norm = plt.Normalize(
            vmin=np.min(curvature_values), vmax=np.max(curvature_values)
        )
        # Apply colormap to normalized values
        edge_colors = [curvature_cmap(norm(value)) for value in curvature_values]

        nx.draw(
            self.graph,
            pos=self.pos,
            nodelist=self.graph.nodes(),
            node_color=mapped_node_color,
            node_size=130,
            edge_color=edge_colors,
            alpha=0.9,
        )
        self.save_and_show()

    def draw_communities(self, clustering_label, nodes_cmap=default_nodes_cmap):
        """
        Draws the communities (identified as the connected components) in subplots.

        :param clustering_label: Node attribute name for clustering (community) labels.
        :type clustering_label: str
        :param nodes_cmap: The colormap for nodes. Default is `default_nodes_cmap`.
        :type nodes_cmap: str, optional
        """
        complex_list = nx.get_node_attributes(self.graph, clustering_label)
        le = preprocessing.LabelEncoder()
        graph_node_color = le.fit_transform(list(complex_list.values()))
        graph_dict_colors = dict(zip(self.graph.nodes(), graph_node_color))

        # Convert the generator to a list to get the connected components
        cc = nx.connected_components(self.graph)
        num_components = len(list(cc))
        print(f"\nDetected {num_components} communities")
        if num_components > 10:
            print("Detected communities are too many for visualization.")
            return 0

        # Create a figure with subplots
        fig, axes = plt.subplots(1, num_components, figsize=(5 * num_components, 5))
        if num_components == 1:
            axes = [axes]  # Ensure axes is iterable if there's only one component

        cc = nx.connected_components(self.graph)  # Get the generator again
        # Plot each connected component in a subplot
        for idx, component in enumerate(cc):
            ax = axes[idx]
            subgraph = self.graph.subgraph(component)
            subgraph_dict_colors = {
                k: graph_dict_colors[k]
                for k in subgraph.nodes()
                if k in graph_dict_colors
            }

            colormap = mcm.get_cmap(nodes_cmap)  # Discrete colormap
            # Create node colors based on community assignment
            node_colors = [colormap(comm) for comm in subgraph_dict_colors.values()]

            curvature_values = [
                d["ricciCurvature"] for _, _, d in self.graph.edges(data=True)
            ]
            # Normalize curvature values between 0 and 1
            norm = plt.Normalize(
                vmin=np.min(curvature_values), vmax=np.max(curvature_values)
            )
            # Apply colormap to normalized values
            edge_colors = [curvature_cmap(norm(value)) for value in curvature_values]

            # Draw the subgraph for the current component
            nx.draw(
                subgraph,
                pos=self.pos,
                nodelist=subgraph.nodes(),
                node_color=node_colors,
                node_size=130,
                edge_color=edge_colors,
                alpha=0.9,
                ax=ax,  # Specify the axis for the current subplot
            )
            ax.set_title(f"Community {idx + 1}")
            ax.axis("off")

        plt.tight_layout()
        self.save_and_show()

    def plot_graph_histo(self, curvature="ricciCurvature"):
        """
        Plots histograms for Ricci curvature and edge weights.

        :param curvature: The edge attribute name for Ricci curvature values. Default is "ricciCurvature".
        :type curvature: str, optional
        """
        # Plot the histogram of Ricci curvatures
        plt.subplot(2, 1, 1)
        ricci_curvtures = nx.get_edge_attributes(self.graph, curvature).values()
        plt.hist(ricci_curvtures, bins=20)
        plt.xlabel("Ricci curvature")
        plt.ylabel("# of nodes")
        plt.title("Histogram of Ricci Curvatures " + self.title)

        # Plot the histogram of edge weights
        plt.subplot(2, 1, 2)
        weights = nx.get_edge_attributes(self.graph, "weight").values()
        plt.hist(weights, bins=20)
        plt.xlabel("Edge weight")
        plt.ylabel("# of nodes")
        plt.title("Histogram of Edge weights " + self.title)

        plt.tight_layout()
        self.save_and_show(plot_axis=True)


def plot_accuracy(maxw, cutoff_range, modularity, ari, save_path, good_cut=None):
    """
    Plots the accuracy of the edge weight cutoff with respect to modularity and Adjusted Rand Index (ARI).

    :param maxw: Maximum edge weight for the x-axis limit.
    :type maxw: float
    :param cutoff_range: Range of edge weight cutoff values.
    :type cutoff_range: list or array of float
    :param modularity: Modularity values corresponding to the cutoff range.
    :type modularity: list or array of float
    :param ari: Adjusted Rand Index values corresponding to the cutoff range.
    :type ari: list or array of float
    :param save_path: Path to save the resulting plot image.
    :type save_path: str
    :param good_cut: Optional edge weight cutoff value that represents a "good" cut. If provided, a vertical line will be drawn at this value. Default is None.
    :type good_cut: float, optional
    """
    plt.xlim(maxw, 0)
    plt.xlabel("Edge weight cutoff")
    plt.plot(cutoff_range, modularity, alpha=0.8)
    plt.plot(cutoff_range, ari, alpha=0.8)

    if good_cut == None:
        plt.legend(["Modularity", "Adjust Rand Index"])
    else:
        plt.axvline(x=good_cut, color="red")
        plt.legend(["Modularity", "Adjust Rand Index", "Good cut"])

    plt.gcf().canvas.manager.set_window_title("Surgery Accuracy")
    plt.savefig(os.path.join(save_path, "Surgery Accuracy.png"), dpi=600)
    plt.show()


def plot_comp_histo(modularity_values, ari_values, save_path):
    """
    Plot and compare modularity and Adjusted Rand Index (ARI) across different community detection methods.

    This function creates two bar charts to visualize the performance of three community detection methods
    (**Ricci Flow**, **Louvain**, and **Girvan-Newman**) based on:

    - **Modularity**: Measures the strength of community structure in the network.
    - **Adjusted Rand Index (ARI)**: Measures clustering accuracy compared to the ground truth.

    The function saves the resulting comparison plot to the specified directory.

    :param modularity_values: Modularity scores for Ricci Flow, Louvain, and Girvan-Newman.
    :type modularity_values: list[float]
    :param ari_values: ARI scores for Ricci Flow, Louvain, and Girvan-Newman.
    :type ari_values: list[float]
    :param save_path: Directory where the comparison plot will be saved.
    :type save_path: str
    """
    # Labels and corresponding values
    methods = ["Ricci Flow", "Louvain", "Girvan-Newman"]

    # X positions for bars
    x = np.arange(len(methods))
    bar_width = 0.4

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Modularity histogram
    axes[0].bar(
        x, modularity_values, width=bar_width, color="b", alpha=0.8, label="Modularity"
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=45)
    axes[0].set_title("Modularity Comparison")
    axes[0].set_ylabel("Modularity")
    axes[0].legend()

    # ARI histogram
    axes[1].bar(x, ari_values, width=bar_width, color="r", alpha=0.8, label="ARI")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=45)
    axes[1].set_title("ARI Comparison")
    axes[1].set_ylabel("ARI")
    axes[1].legend()

    # Adjust layout and show plot
    plt.tight_layout()
    plt.gcf().canvas.manager.set_window_title("Comparison with different methods")
    plt.savefig(os.path.join(save_path, "Comparison.png"), dpi=600)
    plt.show()
