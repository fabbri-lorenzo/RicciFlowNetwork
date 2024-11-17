import networkx as nx
import matplotlib.pyplot as plt
from utils import (
    GraphDrawer,
    compute_ricci_flow,
    perform_surgery,
    detect_communities,
)


def create_SBM_graph():
    sizes = [50, 50]
    p_matrix = [[0.8, 0.05], [0.05, 0.8]]
    G = nx.stochastic_block_model(sizes, p_matrix)
    return G


def create_barbell_graph():
    G = nx.barbell_graph(20, 5)
    return G


def create_caveman_graph():
    G = nx.caveman_graph(5, 10)
    return G


def test_compute_ricci_flow(G):
    print("Before Ricci Flow:")
    GraphDrawer(G).basic_draw(title="Before Ricci Flow")

    # Apply Ricci Flow
    ricci_graph = compute_ricci_flow(G)

    print("After Ricci Flow:")
    GraphDrawer(ricci_graph).draw_with_curvatures(title="After Ricci Flow")

    # Print edge weights and Ricci curvatures
    for u, v, d in ricci_graph.edges(data=True):
        print(
            f"Edge ({u}, {v}): Weight = {d['weight']:.3f}, Ricci Curvature = {d.get('ricciCurvature', 'N/A')}"
        )
    return ricci_graph


def test_perform_surgery(G):
    try:
        user_threshold = float(input("\nThreshold for surgery: "))
    except ValueError:
        print("The inserted value for threshold is not a floating point number.")
    # Perform surgery, removing edges with weight > threshold
    print("Performing surgery on edges...")
    pruned_graph = perform_surgery(G, threshold=user_threshold)

    print("After Surgery:")
    GraphDrawer(pruned_graph).draw_with_curvatures(title="After Surgery")
    return pruned_graph


def test_community_visualization(G, communities):
    GraphDrawer(G).draw_with_communities(communities, title="Communities")


def run_tests():
    """Run all tests on the toy model."""

    try:
        graph_type = int(
            input(
                "\n1 - Stochastic Block Model graph \n2 - Barbell graph \n3 - Caveman graph"
                "\n\nInsert the number corrensponding to the type of graph you would like to have as a test: "
            )
        )
    except ValueError:
        print("The inserted value is not an integer.")
    except graph_type != (1, 2, 3):
        print("The inserted value must be one of 1,2,3.")

    if graph_type == 1:
        G = create_SBM_graph()
    elif graph_type == 2:
        G = create_barbell_graph()
    elif graph_type == 3:
        G = create_caveman_graph()

    # Test Ricci Flow computation
    ricci_graph = test_compute_ricci_flow(G)

    # Perform surgery on graph based on edge weights
    pruned_graph = test_perform_surgery(ricci_graph)

    # Detect communities after Ricci Flow
    communities = detect_communities(pruned_graph)

    # Visualize the graph communities after Ricci Flow and surgery
    test_community_visualization(pruned_graph, communities)


# Run all the tests
if __name__ == "__main__":
    run_tests()
