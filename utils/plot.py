# Standard library imports
import numpy as np
import os

# Third-party imports
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn import preprocessing

# Custom colormap for edge's curvature
colors = [(0, "blue"), (0.5, "black"), (1, "red")]  # Blue (-), Black (0), Red (+)
curvature_cmap = mcolors.LinearSegmentedColormap.from_list("CustomMap", colors)


class GraphDrawer:
    def __init__(self, graph, title, save_path, seed=42):
        self.graph = graph
        self.title = title
        self.save_path = save_path
        self.pos = nx.spring_layout(graph, center=(0, 0), seed=seed)

    def draw_colorbar(self, curvature_values):
        sm = plt.cm.ScalarMappable(
            cmap=curvature_cmap,
            norm=plt.Normalize(vmin=min(curvature_values), vmax=max(curvature_values)),
        )
        plt.colorbar(sm, ax=plt.gca(), label="Ricci Curvature")

    def save_and_show(self, plot_axis=False):
        plt.gcf().canvas.manager.set_window_title(self.title)
        if not plot_axis:
            plt.axis("off")
        plt.savefig(os.path.join(self.save_path, self.title + ".png"), dpi=600)
        plt.show()

    # -----------------------------------

    def draw_graph(self, clustering_label, nodes_cmap="rainbow"):
        """
        A helper function to draw a nx graph with community.
        """
        complex_list = nx.get_node_attributes(self.graph, clustering_label)
        le = preprocessing.LabelEncoder()
        node_color = le.fit_transform(list(complex_list.values()))

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
            node_color=node_color,
            node_size=130,
            cmap=nodes_cmap,
            edge_color=edge_colors,
            alpha=0.9,
        )
        self.save_and_show()

    def draw_communities(self, clustering_label, nodes_cmap="rainbow"):
        complex_list = nx.get_node_attributes(self.graph, clustering_label)
        le = preprocessing.LabelEncoder()
        full_node_color = le.fit_transform(
            list(complex_list.values())
        )  # Full graph coloring

        # Convert the generator to a list to get the connected components
        connected_components = list(nx.connected_components(self.graph))
        num_components = len(connected_components)  # Now we can compute the length

        # Create a figure with subplots
        fig, axes = plt.subplots(1, num_components, figsize=(5 * num_components, 5))
        if num_components == 1:
            axes = [axes]  # Ensure axes is iterable if there's only one component

        # Plot each connected component in a subplot
        for idx, component in enumerate(
            connected_components
        ):  # No need for unpacking here
            ax = axes[idx]
            subgraph = self.graph.subgraph(component)
            # Map the `node_color` array to match only the nodes in the current component
            component_node_list = list(component)
            component_node_color = [
                full_node_color[list(self.graph.nodes).index(node)]
                for node in component_node_list
            ]

            # Draw the subgraph for the current component
            nx.draw(
                subgraph,
                pos=self.pos,
                nodelist=component_node_list,  # Use only nodes in the current component
                node_color=component_node_color,  # Use the filtered node colors
                node_size=130,
                cmap=nodes_cmap,
                alpha=0.9,
                ax=ax,  # Specify the axis for the current subplot
            )
            ax.set_title(f"Community {idx + 1}")
            ax.axis("off")

        plt.tight_layout()
        self.save_and_show()

    def plot_graph_histo(self, curvature="ricciCurvature"):
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


def plot_accuracy(maxw, cutoff_range, modularity, ari, save_path):

    plt.xlim(maxw, 0)
    plt.xlabel("Edge weight cutoff")
    plt.plot(cutoff_range, modularity, alpha=0.8)
    plt.plot(cutoff_range, ari, alpha=0.8)

    plt.legend(["Modularity", "Adjust Rand Index"])

    plt.gcf().canvas.manager.set_window_title("Surgery Accuracy")
    plt.savefig(os.path.join(save_path, "Surgery Accuracy.png"), dpi=600)
    plt.show()
