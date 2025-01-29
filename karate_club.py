from GraphRicciCurvature.OllivierRicci import OllivierRicci
import matplotlib.colors as mcolors
import networkx as nx
from utils.plot import GraphDrawer, plot_accuracy
from utils.surgery import check_accuracy, perform_surgery

# Custom colormap for Karate club nodes
node_colors = ["purple", "orange"]
nodes_cmap = mcolors.ListedColormap(node_colors)


def karate_club():
    """
    This function runs a series of tests on Zachary's Karate Club graph (see https://doi.org/10.1086/jar.33.4.3629752) to compute the Ollivier-Ricci curvature, perform Ricci flow, and evaluate the modularity and accuracy after surgery.

    Steps:
    1. Load the Karate Club graph.
    2. Compute the Ollivier-Ricci curvature of the graph.
    3. Apply Ricci flow to the graph.
    4. Compute modularity and ARI (Adjusted Rand Index) based on the cutoff parameter.
    5. Perform edge surgery based on a user-defined threshold (i.e. the chosen cutoff).
    6. Detect communities as connected components of the resulting graph.
    7. Draw and save visualizations for each of the steps.

    The resulting images and accuracy plots are saved to a directory called 'KarateClubResults'.
    """
    print("\n- Import karate club graph")
    G = nx.karate_club_graph()

    save_path = "KarateClubResults"

    # -----------------------------------
    print("\n=====  Compute Ricci flow metric - Optimal Transportation Distance =====")
    orc = OllivierRicci(G, alpha=0.5, method="OTD")

    orc.compute_ricci_curvature()
    G_orc = orc.G.copy()  # save an intermediate result

    GraphDrawer(G_orc, "Before Ricci Flow", save_path).plot_graph_histo()
    GraphDrawer(G_orc, "Before Ricci Flow (graph)", save_path).draw_graph(
        clustering_label="club", nodes_cmap=nodes_cmap
    )

    # -----------------------------------
    print("\n=====  Perform Ricci flow =====")

    orc.compute_ricci_flow(iterations=50)
    G_rf = orc.G.copy()

    GraphDrawer(G_rf, "After Ricci Flow", save_path).plot_graph_histo()

    GraphDrawer(G_rf, "After Ricci Flow (graph)", save_path).draw_graph(
        clustering_label="club", nodes_cmap=nodes_cmap
    )

    # -----------------------------------
    print("\n=====  Compute Modularity & ARI vs cutoff =====")
    maxw, cutoff_range, modularity, ari, good_cut = check_accuracy(
        G_rf, clustering_label="club", eval_cut=True
    )
    plot_accuracy(maxw, cutoff_range, modularity, ari, save_path, good_cut)

    try:
        user_threshold = float(input("\nThreshold for surgery: "))
    except ValueError:
        print("The inserted value for threshold is not a floating point number.")

    print("\n-  Apply surgery\n")
    G_srg = perform_surgery(G_rf, clustering_label="club", cut=user_threshold)

    GraphDrawer(
        G_srg,
        "After Surgery",
        save_path,
    ).draw_graph(clustering_label="club", nodes_cmap=nodes_cmap)

    # -----------------------------------
    print("\n- Draw communities")
    GraphDrawer(G_srg, "Detected Communities", save_path).draw_communities(
        clustering_label="club", nodes_cmap=nodes_cmap
    )


if __name__ == "__main__":
    karate_club()
