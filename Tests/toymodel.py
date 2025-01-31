from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.getcwd(), "/Users/lorenzofabbri/Downloads/Code/RicciFlowNetwork"
        )  # Substitute with your own path
    )
)
from utils.plot import GraphDrawer, plot_accuracy
from utils.surgery import check_accuracy, perform_surgery


def create_SBM_graph():
    """
    Create a Stochastic Block Model (SBM) graph with 2 equal size communities.

    The sizes of the communities are predefined as 250, 250 (as it is in "Community Detection on Networks with Ricci Flow" by by Chien-Chun Ni et al.). A probability matrix defines
    the edge probabilities within and across the communities. This function also assigns community
    labels to the nodes in the graph.

    :returns: A NetworkX graph with community labels assigned to nodes.
    :rtype: networkx.Graph
    """
    sizes = [250, 250]
    p_matrix = [[0.2, 0.03], [0.03, 0.2]]
    G = nx.stochastic_block_model(sizes, p_matrix)

    # Assign "community" labels to nodes
    start = 0
    for i, size in enumerate(sizes):
        for node in range(start, start + size):
            G.nodes[node]["community"] = f"{i}"
        start += size

    return G


def create_LFR_graph():
    """
    Create an Lancichinetti Fortunato Radicchi (LFR) benchmark graph.

    The graph has 500 nodes with specific degree and community size distributions.
    Community labels are then assigned to the nodes.

    :returns: A NetworkX graph with community labels assigned to nodes.
    :rtype: networkx.Graph
    """
    G = nx.LFR_benchmark_graph(
        n=500,  # Number of nodes
        tau1=3,  # Degree distribution exponent
        tau2=1.5,  # Community size distribution exponent
        mu=0.2,  # Low mixing parameter for strong community structure
        min_community=20,  # Minimum number of nodes in each community
        max_community=70,  # Maximum number of nodes in each community
        average_degree=20,  # Average degree per node
        max_degree=50,  # Maximum degree per node
        max_iters=1000,  # Maximum number of iterations for graph generation
        seed=42,  # Random seed for reproducibility
    )

    complex_list = nx.get_node_attributes(G, "community")
    for node, value in complex_list.items():
        if isinstance(value, set):
            complex_list[node] = str(value)

    nx.set_node_attributes(G, complex_list, "community")

    return G


def test_ricci_curvature(G):
    """
    Compute Ricci curvature of the given graph using Ollivier-Ricci method.

    This function initializes the Ollivier-Ricci curvature calculation on the input graph and computes
    the Ricci curvature using the Optimal Transport Distance (OTD) method.

    :param G: The graph to compute the Ricci curvature on.
    :type G: networkx.Graph
    :returns: The OllivierRicci instance containing computed Ricci curvature.
    :rtype: GraphRicciCurvature.OllivierRicci
    """
    print("\n=====  Before Ricci Flow =====")
    orc = OllivierRicci(G, alpha=0.5, base=1, exp_power=0, proc=4, method="OTD")
    orc.compute_ricci_curvature()

    return orc


def test_ricci_flow(orc, iterations):
    """
    Compute Ricci flow of the graph using Ollivier-Ricci method.

    This function applies the Ricci flow algorithm to the graph and updates the graph's curvature.

    :param orc: The OllivierRicci instance that has the initial Ricci curvature.
    :type orc: GraphRicciCurvature.OllivierRicci
    :returns: The updated graph after applying Ricci flow.
    :rtype: networkx.Graph
    """
    print("\n=====  Compute Ricci flow metric - Optimal Transportation Distance =====")
    orc.compute_ricci_flow(iterations)

    return orc.G


def test_check_accuracy(G):
    """
    Compute Modularity and Adjusted Rand Index (ARI) for different edge weight cutoffs.

    This function tests the community detection performance by checking modularity and ARI
    for different cutoff values applied to the graph.

    :param G: The graph on which the accuracy is tested.
    :type G: networkx.Graph
    :returns: Maximum weight, cutoff range, modularity, and ARI values.
    :rtype: tuple
    """
    print("\n=====  Compute Modularity & ARI vs cutoff =====")
    maxw, cutoff_range, modularity, ari = check_accuracy(
        G, clustering_label="community"
    )

    return maxw, cutoff_range, modularity, ari


def test_perform_surgery(G):
    """
    Perform edge surgery on the graph by removing edges with weight greater than a given threshold.

    The user is prompted to input a threshold value, and edges with weights greater than this threshold
    are removed from the graph.

    :param G: The graph on which the surgery is performed.
    :type G: networkx.Graph
    """
    print("\n=====  After Surgery =====")
    try:
        user_threshold = float(input("\nThreshold for surgery: "))
    except ValueError:
        print("The inserted value for threshold is not a floating point number.")
    # Perform surgery, removing edges with weight > threshold
    print("Performing surgery on edges...")
    perform_surgery(G, cut=user_threshold)


def run_tests():
    """
    Run all tests including graph generation, Ricci curvature computation,
    accuracy checking, and surgery performance.

    The function allows the user to choose between generating a Stochastic Block Model (SBM) graph
    or an LFR benchmark graph. It then computes the Ricci curvature, performs Ricci flow, checks
    modularity and ARI, and performs edge surgery on the graph.

    :returns: None
    :rtype: None
    """
    try:
        graph_type = int(
            input(
                "\n1 - Stochastic Block Model graph \n2 - LFR Benchmark graph"
                "\n\nInsert the number corresponding to the type of graph you would like to have as a test: "
            )
        )
        if graph_type not in (1, 2):
            print("The inserted value must be 1 or 2")
            return -1
    except ValueError:
        print("The inserted value is not an integer.")

    if graph_type == 1:
        G = create_SBM_graph()
        save_path = "tests/ToyModelResults/SBM"
        iterations = 10

    elif graph_type == 2:
        G = create_LFR_graph()
        save_path = "tests/ToyModelResults/LFR"
        iterations = 40
    # -----------------------------------
    orc = test_ricci_curvature(G)
    GraphDrawer(orc.G, "Before Ricci Flow", save_path).draw_graph(
        clustering_label="community"
    )
    # -----------------------------------
    G_rf = test_ricci_flow(orc, iterations)
    GraphDrawer(G_rf, "After Ricci Flow", save_path).draw_graph(
        clustering_label="community"
    )
    # -----------------------------------
    maxw, cutoff_range, modularity, ari = test_check_accuracy(G_rf)
    plot_accuracy(maxw, cutoff_range, modularity, ari, save_path)
    # -----------------------------------
    test_perform_surgery(G_rf)
    GraphDrawer(G_rf, "After Surgery", save_path).draw_graph(
        clustering_label="community"
    )
    # -----------------------------------
    print("\n- Drawing communities")
    GraphDrawer(G_rf, "Detected Communities", save_path).draw_communities(
        clustering_label="community"
    )


if __name__ == "__main__":
    run_tests()
