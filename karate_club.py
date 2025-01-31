from GraphRicciCurvature.OllivierRicci import OllivierRicci
import matplotlib.colors as mcolors
import networkx as nx
from utils.plot import GraphDrawer, plot_accuracy, plot_comp_histo
from utils.surgery import ARI, check_accuracy, perform_surgery

# Custom colormap for Karate club nodes
node_colors = ["purple", "orange"]
nodes_cmap = mcolors.ListedColormap(node_colors)


def karate_club_rf():
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

    :returns: A tuple containing:
        - **best_mod** (*float*): Modularity value corrensponding to highest ari.
        - **best_ari** (*float*): Highest Adjusted Rand Index obtained applying Ricci Flow.
    :rtype: tuple(float, float)
    """
    print("\n- Import karate club graph")
    G = nx.karate_club_graph()

    save_path = "KarateClubResults"
    # -----------------------------------

    print("\n=====  Compute Ricci flow metric - Optimal Transportation Distance =====")
    orc = OllivierRicci(G, alpha=0.5, method="OTD")

    orc.compute_ricci_curvature()

    GraphDrawer(orc.G, "Before Ricci Flow", save_path).plot_graph_histo()
    GraphDrawer(orc.G, "Before Ricci Flow (graph)", save_path).draw_graph(
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
    maxw, cutoff_range, modularity, ari, good_cut, best_ari, best_mod = check_accuracy(
        G_rf, clustering_label="club", eval_cut=True
    )
    plot_accuracy(maxw, cutoff_range, modularity, ari, save_path, good_cut)

    try:
        user_threshold = float(input("\nThreshold for surgery: "))
    except ValueError:
        print("The inserted value for threshold is not a floating point number.")

    print("\n-  Apply surgery\n")
    perform_surgery(G_rf, clustering_label="club", cut=user_threshold)

    GraphDrawer(
        G_rf,
        "After Surgery",
        save_path,
    ).draw_graph(clustering_label="club", nodes_cmap=nodes_cmap)
    # -----------------------------------

    print("\n- Draw communities")
    GraphDrawer(G_rf, "Detected Communities", save_path).draw_communities(
        clustering_label="club", nodes_cmap=nodes_cmap
    )

    return best_mod, best_ari


import community.community_louvain as community_louvain  # Louvain method
import igraph as ig
import leidenalg as la  # Leiden method
from networkx.algorithms.community import girvan_newman


def karate_club_comp():
    """
    Compare Louvain and Girvan-Newman community detection on the Zachary Karate Club graph.

    This function applies **Louvain** and **Girvan-Newman** algorithms to the
    Zachary Karate Club graph to detect communities. It evaluates both methods using:

    - **Modularity**: Measures the strength of the community structure.
    - **Adjusted Rand Index (ARI)**: Measures clustering accuracy compared to the ground truth.

    The function prints the modularity and ARI scores for both methods and returns them.

    :returns: A tuple containing:
        - **louvain_modularity** (*float*): Modularity score for Louvain clustering.
        - **louvain_ari** (*float*): ARI score for Louvain clustering.
        - **gn_modularity** (*float*): Modularity score for Girvan-Newman clustering.
        - **gn_ari** (*float*): ARI score for Girvan-Newman clustering.
    :rtype: tuple(float, float, float, float)
    """
    G = nx.karate_club_graph()

    # --- Louvain ---
    louvain_partition = community_louvain.best_partition(G)
    louvain_communities = {
        frozenset([node for node in louvain_partition if louvain_partition[node] == c])
        for c in set(louvain_partition.values())
    }
    louvain_modularity = nx.algorithms.community.quality.modularity(
        G, louvain_communities
    )
    louvain_ari = ARI(G, louvain_partition, "club")

    # --- Girvan-Newman ---
    gn_hierarchy = girvan_newman(G)
    gn_partition = next(gn_hierarchy)  # First split
    gn_modularity = nx.algorithms.community.quality.modularity(
        G, list(map(set, gn_partition))
    )
    gn_ari = ARI(G, gn_partition, "club")

    print(
        f"\nLouvain Modularity: {louvain_modularity:.3f}, Louvain ARI: {louvain_ari:.3f}"
    )
    print(
        f"\nGirvan-Newman Modularity: {gn_modularity:.3f}, Girvan-Newman ARI: {gn_ari:.3f}"
    )

    return (louvain_modularity, louvain_ari, gn_modularity, gn_ari)


if __name__ == "__main__":
    modularity_rf, ari_rf = karate_club_rf()
    (louvain_modularity, louvain_ari, gn_modularity, gn_ari) = karate_club_comp()

    modularity_values = [modularity_rf, louvain_modularity, gn_modularity]
    ari_values = [ari_rf, louvain_ari, gn_ari]

    plot_comp_histo(modularity_values, ari_values, "KarateClubResults")
