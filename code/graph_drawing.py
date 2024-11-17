# Standard library imports
import numpy as np

# Third-party imports
import networkx as nx
import matplotlib.pyplot as plt

# Import from configuration file
import config as cfg


class GraphDrawer:
    def __init__(self, graph, seed=cfg.seed_value):
        self.graph = graph
        # Use a spring layout with lower k for a more circular spread
        self.pos = nx.spring_layout(
            graph, k=3.5, iterations=600, center=(0, 0), seed=seed
        )

    def draw_nodes(self, node_color=(0.64, 0.64, 0.82), size=30, alpha=1.0):
        # If a single color (tuple) is provided, repeat it for all nodes
        if isinstance(node_color, tuple) and len(node_color) == 3:
            node_color = [node_color] * self.graph.number_of_nodes()

        nx.draw_networkx_nodes(
            self.graph,
            self.pos,
            node_size=size,
            node_color=node_color,  # Ensure node_color is a list of colors
            alpha=alpha,
        )

    def draw_edges(self, edge_values, widths=None, cmap=plt.cm.Blues):
        # Normalize edge values between 0 and 1
        norm = plt.Normalize(vmin=np.min(edge_values), vmax=np.max(edge_values))

        # Apply colormap to normalized values
        edge_colors = [cmap(norm(value)) for value in edge_values]

        # Draw edges with specified colors
        nx.draw_networkx_edges(
            self.graph,
            self.pos,
            edge_color=edge_colors,
            width=widths if widths else 1.0,
        )

    def basic_draw(self, node_color=(0.64, 0.64, 0.82), title="Graph", show=True):

        # Draw the graph
        plt.figure(figsize=(10, 8))
        # Draw nodes and edges using the helper methods
        self.draw_nodes(node_color=node_color, size=30, alpha=1.0)  # Use `node_color`
        nx.draw_networkx_edges(self.graph, self.pos, edge_color="lightblue")

        plt.gcf().canvas.manager.set_window_title(title)
        plt.axis("off")
        # Save with high DPI (if needed)
        plt.savefig(title + ".png", dpi=600)
        if show:
            plt.show()

    def draw_with_curvatures(
        self, node_color=(0.64, 0.64, 0.82), title="Graph with Edge Labels", show=True
    ):
        curvature_values = [
            d["ricciCurvature"] for u, v, d in self.graph.edges(data=True)
        ]
        edge_widths = [0.5 + abs(curvature) * 1.5 for curvature in curvature_values]

        # Draw nodes and edges using the helper methods
        self.draw_nodes(node_color=node_color, size=30, alpha=0.8)  # Use `node_color`
        self.draw_edges(
            edge_values=curvature_values, widths=edge_widths, cmap=plt.cm.RdBu
        )

        # Add color bar for Ricci Curvature
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.RdBu,
            norm=plt.Normalize(vmin=min(curvature_values), vmax=max(curvature_values)),
        )
        plt.colorbar(sm, ax=plt.gca(), label="Ricci Curvature")

        plt.gcf().canvas.manager.set_window_title(title)
        plt.axis("off")
        # Save with high DPI (if needed)
        plt.savefig(title + ".png", dpi=600)
        if show:
            plt.show()

    def draw_with_communities(self, communities, title="Graph with Communities"):
        community_colors = [communities[node] for node in self.graph.nodes()]

        if len(community_colors) != len(self.graph.nodes()):
            raise ValueError(
                f"Number of colors ({len(community_colors)}) does not match number of nodes ({len(self.graph.nodes())})"
            )

        # Map community indices to colors using a colormap
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(vmin=min(community_colors), vmax=max(community_colors))
        node_color_mapped = [cmap(norm(community)) for community in community_colors]

        # Call draw_with_curvatures with the mapped node colors
        self.draw_with_curvatures(node_color=node_color_mapped, title=title)
