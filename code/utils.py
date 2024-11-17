# Standard library imports
import numpy as np
from collections import defaultdict
import pickle, random


# Third-party imports
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain  # Louvain method for community detection

# Project-specific imports
from GraphRicciCurvature.OllivierRicci import OllivierRicci

# Import from configuration file
import config as cfg


def generate_planar_graph_with_communities(
    num_communities, nodes_in_communities, title="graph"
):
    # Create an empty planar graph
    G = nx.Graph()
    community_labels = {}
    node_colors = []
    # Add nodes and edges to the graph
    for community in range(num_communities):
        # Create nodes for each community
        start_node = 0 if community == 0 else sum(nodes_in_communities[:(community)])
        end_node = start_node + nodes_in_communities[community]
        G.add_nodes_from(range(start_node, end_node))
        print(f"start node: {start_node}, end node: {end_node}\n")
        # Add edges within the community
        for node1 in range(start_node, end_node):
            for node2 in range(start_node, end_node):
                if node1 != node2 and random.random() < (
                    10 / len(range(start_node, end_node))
                ):
                    G.add_edge(node1, node2)
                    # Ensure the graph is planar (remove edges if necessary)
                    if not nx.check_planarity(G)[0]:
                        G.remove_edge(node1, node2)

            node_colors.append(cfg.community_colors_data[community])
            community_labels[node1] = community

    # Add some inter-community nodes
    total_nodes = sum(nodes_in_communities[:(num_communities)])
    for node1 in range(total_nodes):
        for node2 in range(total_nodes):
            if node1 != node2 and random.random() < (
                0.1 / len(cfg.nodes_in_communities)
            ):
                G.add_edge(node1, node2)
                if not nx.check_planarity(G)[0]:
                    G.remove_edge(node1, node2)

    # Save the graph to a file
    with open(title, "wb") as f:
        pickle.dump((G, community_labels, node_colors), f)

    return G, community_labels, node_colors


def compute_ricci_flow(graph, iterations=15, alpha=0.5):
    orc = OllivierRicci(graph, alpha=alpha)
    orc.compute_ricci_curvature()

    # Manually assign the Ricci curvature to the edges
    for u, v, curvature in orc.G.edges(data="ricciCurvature"):
        if curvature is not None:  # Ensure curvature exists
            graph[u][v]["ricciCurvature"] = curvature
            # if curvature < 0:
            # print(f"{[u]}{[v]} curvature is {curvature}\n")
    # Simulate Ricci flow process
    for i in range(iterations):
        for u, v, d in graph.edges(data=True):
            curvature = d.get(
                "ricciCurvature", 0
            )  # Handle missing curvature gracefully
            weight_update = curvature * d.get("weight", 1)
            d["weight"] = max(d.get("weight", 1) + weight_update, 0.1)

    return graph


def perform_surgery(graph, threshold=0):
    edges_to_remove = [
        (u, v)
        for u, v, d in graph.edges(data=True)
        if d.get("ricciCurvature", 0) < threshold
    ]
    graph.remove_edges_from(edges_to_remove)
    return graph


def apply_ricci_flow_and_surgery(
    graph, iterations=5, rounds=3, threshold=0, increment_sr_th_each_round=0, alpha=0.5
):
    for round_num in range(rounds):

        # print(f"Round {round_num + 1}/{rounds}: Starting Ricci Flow")
        graph = compute_ricci_flow(graph, iterations=iterations, alpha=alpha)
        # print(f"Graph after Ricci Flow: {graph.number_of_edges()} edges")

        # print(f"Round {round_num + 1}/{rounds}: Starting Surgery")
        graph = perform_surgery(
            graph, threshold=threshold + increment_sr_th_each_round * round_num
        )
        # print(f"Graph after Surgery: {graph.number_of_edges()} edges")

    return graph


def detect_communities(G, lenght_threshold=300):
    # calculate lenght of each edge of the graph
    pos = nx.spring_layout(G)
    for u, v in G.edges():
        # Calculate Euclidean distance between nodes u and v
        G[u][v]["length"] = (
            (pos[u][0] - pos[v][0]) ** 2 + (pos[u][1] - pos[v][1]) ** 2
        ) ** 0.5
    edges_to_remove = [
        (u, v) for u, v, d in G.edges(data=True) if G[u][v]["length"] > lenght_threshold
    ]
    G.remove_edges_from(edges_to_remove)

    # Extract connected components as communities
    connected_components = list(nx.connected_components(G))
    communities = {}
    # Assign each node a component label based on its connected component index
    for component_index, component in enumerate(connected_components):
        for node in component:
            communities[node] = component_index

    return G, communities


def get_community_sizes(
    communities,
):  # added to convert the detected community assignments into sizes.
    # Initialize a dictionary to store community sizes
    community_sizes = {}

    # Iterate through the partition dictionary
    for node, community in communities.items():
        # Count the number of nodes in each community
        if community not in community_sizes:
            community_sizes[community] = 0
        community_sizes[community] += 1
    return community_sizes


def get_node_groups_by_community(communities):
    """
    This function groups nodes by their community label.

    Returns a list of sets, where each set contains nodes belonging to the same community.
    """
    # Initialize a dictionary to store node groups by community
    community_groups = {}

    # Iterate through the nodes and their community labels
    for node, community in communities.items():
        # If this community hasn't been seen before, initialize a new set for it
        if community not in community_groups:
            community_groups[community] = set()
        # Add the node to the corresponding community group
        community_groups[community].add(node)

    # Convert the dictionary values (sets of nodes) into a list and return it
    return list(community_groups.values())


def get_labels_from_communities(communities, nodes):
    """Extracts ordered labels for each node in nodes based on community assignments."""
    return [communities[node] for node in nodes]


def adjusted_rand_index(true_labels, predicted_labels):
    """Compute the Adjusted Rand Index between two clusterings."""
    contingency = defaultdict(lambda: defaultdict(int))
    for t, p in zip(true_labels, predicted_labels):
        contingency[t][p] += 1

    # Determine the maximum length of the lists in contingency values
    max_len = max(len(p.values()) for p in contingency.values())

    # Pad each list with zeros to ensure uniform length
    padded_contingency = [
        list(p.values()) + [0] * (max_len - len(p.values()))
        for p in contingency.values()
    ]

    # Now create the NumPy array
    contingency_matrix = np.array(padded_contingency)

    # Compute pair counts
    sum_rows = np.sum(contingency_matrix, axis=1)
    sum_cols = np.sum(contingency_matrix, axis=0)
    n = np.sum(contingency_matrix)
    total_pairs = n * (n - 1) // 2

    sum_combinations_contingency = np.sum(
        [n_ij * (n_ij - 1) // 2 for n_ij in contingency_matrix.flatten()]
    )
    sum_combinations_rows = np.sum([n_i * (n_i - 1) // 2 for n_i in sum_rows])
    sum_combinations_cols = np.sum([n_j * (n_j - 1) // 2 for n_j in sum_cols])

    # Compute ARI
    index = sum_combinations_contingency
    expected_index = sum_combinations_rows * sum_combinations_cols / total_pairs
    max_index = (sum_combinations_rows + sum_combinations_cols) / 2
    if max_index == expected_index:
        return 1.0
    else:
        ari = (index - expected_index) / (max_index - expected_index)
        return ari


def analyze_communities(
    title, detected_communities, actual_communities, community_colors
):
    # Ensure detected and actual node-to-community mappings have the same nodes
    all_nodes = set(detected_communities.keys()).union(set(actual_communities.keys()))

    detected_labels = []
    actual_labels = []

    # For each node, get the community label from detected and actual communities
    for node in all_nodes:
        detected_labels.append(
            detected_communities.get(node, -1)
        )  # -1 for nodes not detected
        actual_labels.append(
            actual_communities.get(node, -1)
        )  # -1 for nodes not in actual

    # Use Adjusted Rand Index (ARI) to compute similarity between two community assignments
    ari_score = adjusted_rand_index(actual_labels, detected_labels)
    print(f"\nAdjusted Rand Index (ARI) for community detection: {ari_score:.2%}")

    # The ARI score is a better measure for comparing how well the clusters (communities) match
    # ARI = 1 indicates perfect agreement, ARI = 0 indicates random assignments, and ARI < 0 means disagreement

    # Plotting is the same as before (comparing sizes):
    detected_sizes = get_community_sizes(detected_communities)
    actual_sizes = get_community_sizes(actual_communities)

    # Prepare the x-ticks for the maximum number of communities
    detected_values = list(detected_sizes.values())
    actual_values = list(actual_sizes.values())
    max_length = max(len(detected_values), len(actual_values))
    x = np.arange(max_length)

    # Extend detected_values and actual_values with zeros if necessary
    detected_values.extend([0] * (max_length - len(detected_values)))
    actual_values.extend([0] * (max_length - len(actual_values)))

    # Set the bar width
    bar_width = 0.35

    # Plot detected community sizes with corresponding colors
    detected_colors = [
        community_colors.get(key, "lightgray") for key in range(max_length)
    ]
    plt.bar(
        x - bar_width / 2,
        detected_values,
        width=bar_width,
        label="Detected Sizes",
        color=detected_colors,
    )

    # Plot actual community sizes with dashed lines in light green
    plt.bar(
        x + bar_width / 2,
        actual_values,
        width=bar_width,
        label="Actual Sizes",
        color=detected_colors,
        alpha=0.5,
        hatch="//",
    )

    plt.xticks(x, [f"{i + 1}" for i in range(max_length)])
    plt.title("Community Size Comparison")
    plt.xlabel("Community")
    plt.ylabel("Size")
    plt.legend()
    # Save with high DPI (if needed)
    plt.savefig(title + ".png", dpi=600)
    plt.show()


def is_converging(ricci_graph0):
    initial_communities = detect_communities(ricci_graph0)
    ricci_graph1 = compute_ricci_flow(ricci_graph0, iterations=5, alpha=0.5)
    communities1 = detect_communities(ricci_graph1)
    ricci_graph2 = compute_ricci_flow(ricci_graph1, iterations=5, alpha=0.5)
    communities2 = detect_communities(ricci_graph2)
    if initial_communities == communities1 == communities2:
        print("\nRicci Flow has converged.")
        return True
    else:
        print("\nRicci Flow has NOT converged.")
        return False
