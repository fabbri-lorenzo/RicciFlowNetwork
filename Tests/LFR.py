from GraphRicciCurvature.OllivierRicci import OllivierRicci
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.getcwd(), "/Users/lorenzofabbri/Downloads/Code/RicciFlowNetwork"
        )  # Substitute with your own path
    )
)
from utils.surgery import ARI, get_best_cut, perform_surgery


def get_ARI_LFR(mu, av_deg):
    """
    Generate an LFR benchmark graph, apply Ricci Flow-based community detection, and compute ARI.

    This function creates a **Lancichinetti-Fortunato-Radicchi (LFR) benchmark graph** with
    given parameters, applies **Ollivier-Ricci curvature** and **Ricci Flow**, and evaluates
    the clustering performance using **Adjusted Rand Index (ARI)**.

    :param mu: Mixing parameter (fraction of links connecting different communities).
    :type mu: float
    :param av_deg: Average degree of nodes in the graph.
    :type av_deg: int
    :returns: Adjusted Rand Index (ARI) for the detected communities.
    :rtype: float
    """
    print("\n\n- Generating LFR graph: ", end="")

    # dicts with average degrees as keys and graph parameters as values
    min_com = {20: 20, 25: 30, 30: 40, 35: 50, 40: 60, 45: 70}
    max_com = {20: 40, 25: 50, 30: 65, 35: 65, 40: 80, 45: 90}
    max_deg = {20: 45, 25: 60, 30: 75, 35: 80, 40: 90, 45: 105}

    G = nx.LFR_benchmark_graph(
        n=1000,  # Number of nodes
        tau1=2.5,  # Degree distribution exponent
        tau2=1.8,  # Community size distribution exponent
        mu=mu,  # Mixing parameter
        min_community=min_com.get(av_deg),  # Min num of nodes in each community
        max_community=max_com.get(av_deg),  # Max num of nodes in each community
        average_degree=av_deg,  # Average degree per node
        max_degree=max_deg.get(av_deg),  # Maximum degree per node
        max_iters=1000,  # Maximum number of iterations allowed to generate the graph
        seed=42,  # Random seed for reproducibility
    )
    print("generated. ", end="")
    # -----------------------------------

    complex_list = nx.get_node_attributes(G, "community")
    for node, value in complex_list.items():
        if isinstance(value, set):
            complex_list[node] = str(value)

    nx.set_node_attributes(G, complex_list, "community")
    # -----------------------------------

    print("Computing Ricci Curvature... ", end="")
    orc = OllivierRicci(G, alpha=0.5, method="OTD")
    orc.compute_ricci_curvature()
    # -----------------------------------

    print("Computing Ricci flow... ", end="")
    orc.compute_ricci_flow(iterations=10)
    # -----------------------------------

    print("Performing surgery... ")
    best_cut = get_best_cut(orc.G, clustering_label="community")
    perform_surgery(orc.G, cut=best_cut)
    cc = nx.connected_components(orc.G)
    ari = ARI(orc.G, list(cc), "community")
    print(f"ARI={ari}")

    return ari


def test_LFR():
    """
    See Figure 8(a) from **Ni et al., "Community Detection on Networks with Ricci Flow"**.

    This function tests Ricci Flow-based community detection on LFR benchmark graphs
    for different values of the mixing parameter :math:`\mu`. The results are visualized
    in a plot showing ARI as a function of :math:`\mu` for different average degrees.

    The experiment is run for the following values:
    - :math:`\mu = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]`
    - Average degrees: `[20, 25, 30, 35, 40]`

    The final results are saved as `LFR.png` in the `Tests/LFRResults` directory.
    """
    mu_values = np.arange(0.25, 0.55, 0.05)
    data_avgDeg_20 = [
        get_ARI_LFR(mu, 20)
        for mu in tqdm(mu_values, desc="\nProcessing graph with avgDeg=20")
    ]
    data_avgDeg_25 = [
        get_ARI_LFR(mu, 25)
        for mu in tqdm(mu_values, desc="\nProcessing graph with avgDeg=25")
    ]
    data_avgDeg_30 = [
        get_ARI_LFR(mu, 30)
        for mu in tqdm(mu_values, desc="\nProcessing graph with avgDeg=30")
    ]
    data_avgDeg_35 = [
        get_ARI_LFR(mu, 35)
        for mu in tqdm(mu_values, desc="\nProcessing graph with avgDeg=35")
    ]
    data_avgDeg_40 = [
        get_ARI_LFR(mu, 40)
        for mu in tqdm(mu_values, desc="\nProcessing graph with avgDeg=40")
    ]

    # Plot the data
    plt.plot(
        mu_values,
        data_avgDeg_20,
        label="avgDeg=20",
        marker="s",
        markerfacecolor="none",
        linestyle="-",
        color="blue",
        markersize=10,
    )
    plt.plot(
        mu_values,
        data_avgDeg_25,
        label="avgDeg=25",
        marker="^",
        linestyle="-",
        color="red",
        markersize=10,
    )
    plt.plot(
        mu_values,
        data_avgDeg_30,
        label="avgDeg=30",
        marker="o",
        markerfacecolor="none",
        linestyle="-",
        color="green",
        markersize=10,
    )
    plt.plot(
        mu_values,
        data_avgDeg_35,
        label="avgDeg=35",
        marker="o",
        linestyle="-",
        color="purple",
        markersize=10,
    )
    plt.plot(
        mu_values,
        data_avgDeg_40,
        label="avgDeg=40",
        marker="^",
        markerfacecolor="none",
        linestyle="-",
        color=(0.3, 0.8, 0.8),
        markersize=10,
    )

    # Add labels and title
    plt.xlabel(r"$\mu$")
    plt.ylabel("ARI")
    plt.title(r"Ricci Flow on LFR graph with 1000 nodes - OTD")

    save_path = "Tests/LFRResults"
    plt.legend(loc="upper right", fontsize=14)
    plt.tight_layout()
    plt.gcf().canvas.manager.set_window_title(r"LFR Graph")
    plt.savefig(os.path.join(save_path, "LFR_OTD.png"), dpi=600)
    plt.show()


if __name__ == "__main__":
    test_LFR()
