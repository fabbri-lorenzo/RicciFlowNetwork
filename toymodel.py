import networkx as nx
import matplotlib.colors as mcolors

from GraphRicciCurvature.OllivierRicci import OllivierRicci

# Imports form header files
from utils.plot import GraphDrawer, plot_accuracy
from utils.surgery import perform_surgery, check_accuracy

# Custom colormap for Toy Model nodes
colors = [
    (0, "purple"),
    (0.2, "orange"),
    (0.4, "brown"),
    (0.6, "pink"),
    (0.8, "green"),
    (1, "lightblue"),
]
nodes_cmap = mcolors.LinearSegmentedColormap.from_list("CustomMap", colors)


def create_SBM_graph():
    sizes = [50, 20, 30]  # 3 communities of 50, 20, 30 nodes respectively
    p_matrix = [
        [0.2, 0.03, 0.01],  # probabilities of edges within and across communities
        [0.03, 0.2, 0.02],
        [0.01, 0.02, 0.2],
    ]
    G = nx.stochastic_block_model(sizes, p_matrix)

    # Assign "community" labels to nodes
    start = 0
    for i, size in enumerate(sizes):
        for node in range(start, start + size):
            G.nodes[node]["community"] = f"{i}"
        start += size

    return G


def create_caveman_graph():
    l = 6
    k = 10
    G = nx.caveman_graph(l, k)  # l communities of k nodes each

    # Assign "community" labels to nodes based on their clique
    for node in G.nodes():
        G.nodes[node]["community"] = f"{node // k}"

    return G


def test_ricci_curvature(G):
    print("\n=====  Before Ricci Flow =====")
    orc = OllivierRicci(G, alpha=0.5, method="OTD")
    orc.compute_ricci_curvature()

    return orc


def test_ricci_flow(orc):
    print("\n=====  Compute Ricci flow metric - Optimal Transportation Distance =====")
    orc.compute_ricci_flow(iterations=20)

    return orc.G


def test_check_accuracy(G):
    print("\n=====  Compute Modularity & ARI vs cutoff =====")
    maxw, cutoff_range, modularity, ari = check_accuracy(
        G, clustering_label="community"
    )

    return maxw, cutoff_range, modularity, ari


def test_perform_surgery(G):
    print("\n=====  After Surgery: =====")
    try:
        user_threshold = float(input("\nThreshold for surgery: "))
    except ValueError:
        print("The inserted value for threshold is not a floating point number.")
    # Perform surgery, removing edges with weight > threshold
    print("Performing surgery on edges...")
    G_srg = perform_surgery(G, clustering_label="community", cut=user_threshold)

    return G_srg


def run_tests():
    """Run all tests on the toy model."""
    try:
        graph_type = int(
            input(
                "\n1 - Stochastic Block Model graph \n2 - Caveman graph"
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
        save_path = "ToyModelResults/SBM"

    elif graph_type == 2:
        G = create_caveman_graph()
        save_path = "ToyModelResults/Caveman"
    # -----------------------------------
    orc = test_ricci_curvature(G)
    GraphDrawer(orc.G, "Before Ricci Flow", save_path).draw_graph(
        clustering_label="community", nodes_cmap=nodes_cmap
    )
    # -----------------------------------
    G_rf = test_ricci_flow(orc)
    GraphDrawer(G_rf, "After Ricci Flow", save_path).draw_graph(
        clustering_label="community", nodes_cmap=nodes_cmap
    )
    # -----------------------------------
    maxw, cutoff_range, modularity, ari = test_check_accuracy(G_rf)
    plot_accuracy(maxw, cutoff_range, modularity, ari, save_path)
    # -----------------------------------
    G_srg = test_perform_surgery(G_rf)
    GraphDrawer(G_srg, "After Surgery", save_path).draw_graph(
        clustering_label="community", nodes_cmap=nodes_cmap
    )
    # -----------------------------------
    print("\n- Drawing communities")
    GraphDrawer(G_srg, "Detected Communities", save_path).draw_communities(
        clustering_label="community", nodes_cmap=nodes_cmap
    )


if __name__ == "__main__":
    run_tests()
