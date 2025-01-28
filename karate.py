import networkx as nx
import matplotlib.colors as mcolors

from GraphRicciCurvature.OllivierRicci import OllivierRicci

# Imports form header files
from utils.plot import GraphDrawer, plot_accuracy
from utils.surgery import perform_surgery, check_accuracy

# Custom colormap for Karate club nodes
colors = [(0, "purple"), (1, "orange")]
nodes_cmap = mcolors.LinearSegmentedColormap.from_list("CustomMap", colors)

if __name__ == "__main__":
    print("\n- Import karate club graph")
    G = nx.karate_club_graph()

    save_path = "KarateResults"

    # -----------------------------------
    print("\n=====  Compute Ricci flow metric - Optimal Transportation Distance =====")
    orc = OllivierRicci(G, alpha=0.5, method="OTD")

    orc.compute_ricci_curvature()
    G_orc = orc.G.copy()  # save an intermediate result

    GraphDrawer(G_orc, "Karate graph before RF", save_path).plot_graph_histo()
    GraphDrawer(G_orc, "Karate graph", save_path).draw_graph(
        clustering_label="club", nodes_cmap=nodes_cmap
    )

    # -----------------------------------
    print("\n=====  Perform Ricci flow =====")

    orc.compute_ricci_flow(iterations=50)
    G_rf = orc.G.copy()

    GraphDrawer(G_rf, "Karate graph after RF", save_path).plot_graph_histo()

    GraphDrawer(G_rf, "Karate graph after Ricci Flow", save_path).draw_graph(
        clustering_label="club", nodes_cmap=nodes_cmap
    )

    # -----------------------------------
    print("\n=====  Compute Modularity & ARI vs cutoff =====")
    maxw, cutoff_range, modularity, ari = check_accuracy(G_rf, clustering_label="club")
    plot_accuracy(maxw, cutoff_range, modularity, ari, save_path)

    try:
        user_threshold = float(input("\nThreshold for surgery: "))
    except ValueError:
        print("The inserted value for threshold is not a floating point number.")

    print("\n-  Applying surgery...\n")
    G_srg = perform_surgery(G_rf, clustering_label="club", cut=user_threshold)

    GraphDrawer(
        G_srg,
        "Karate graph after Surgery",
        save_path,
    ).draw_graph(clustering_label="club", nodes_cmap=nodes_cmap)

    # -----------------------------------
    print("\n- Drawing communities")
    GraphDrawer(G_srg, "Karate graph detected communities", save_path).draw_communities(
        clustering_label="club", nodes_cmap=nodes_cmap
    )
