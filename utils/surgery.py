from importlib import util
from sklearn import preprocessing, metrics

import community as community_louvain
import networkx as nx
import numpy as np

dcp = 3  # decimal precision


def ARI(G, clustering, clustering_label):
    """
    Compute the Adjust Rand Index (clustering accuracy) of "clustering" with "clustering_label" as ground truth.

    :param G: A graph with node attribute "clustering_label" as ground truth.
    :type G: networkx.Graph
    :param clustering: Predicted community clustering.
    :type clustering: dict, list, or list of sets
    :param clustering_label: Node attribute name for ground truth.
    :type clustering_label: str
    :returns: Adjust Rand Index for predicted community.
    :rtype: float
    """
    complex_list = nx.get_node_attributes(G, clustering_label)
    le = preprocessing.LabelEncoder()
    y_true = le.fit_transform(list(complex_list.values()))

    if isinstance(clustering, dict):
        # python-louvain partition format
        y_pred = np.array([clustering[v] for v in complex_list.keys()])
    elif isinstance(clustering[0], set):
        # networkx partition format
        predict_dict = {c: idx for idx, comp in enumerate(clustering) for c in comp}
        y_pred = np.array([predict_dict[v] for v in complex_list.keys()])
    elif isinstance(clustering, list):
        # sklearn partition format
        y_pred = clustering
    else:
        return -1

    return metrics.adjusted_rand_score(y_true, y_pred)


def perform_surgery(G_origin: nx.Graph, clustering_label, weight="weight", cut=0):
    """
    A simple surgery function that removes the edges with weight above a threshold.

    :param G_origin: A graph with ``weight`` as the Ricci flow metric to cut.
    :type G_origin: networkx.Graph
    :param weight: The edge weight used as the Ricci flow metric. Defaults to "weight".
    :type weight: str
    :param cut: Manually assigned cutoff point.
    :type cut: float or None
    :returns: A graph after surgery.
    :rtype: networkx.Graph
    """
    G = G_origin.copy()
    w = nx.get_edge_attributes(G, weight)

    assert cut >= 0, "Cut value should be greater than 0."
    if not cut:
        cut = (max(w.values()) - 1.0) * 0.6 + 1.0  # Guess a cut point as default

    to_cut = []
    for n1, n2 in G.edges():
        if G[n1][n2][weight] > cut:
            to_cut.append((n1, n2))
    print("*************** Surgery ****************")
    print("* Cut %d edges." % len(to_cut))
    G.remove_edges_from(to_cut)
    print("* Number of nodes now: %d" % G.number_of_nodes())
    print("* Number of edges now: %d" % G.number_of_edges())
    cc = list(nx.connected_components(G))
    print(
        f"* Modularity now: {nx.algorithms.community.quality.modularity(G, cc):.{dcp}f}"
    )
    print(f"* ARI now: {ARI(G, cc, clustering_label):.{dcp}f}")
    print("****************************************")

    return G


def check_accuracy(G_origin, weight="weight", clustering_label="value", eval_cut=False):
    """
    Check the clustering quality while cutting edges with weight using different thresholds.

    :param G_origin: A graph with ``weight`` as the Ricci flow metric to cut.
    :type G_origin: networkx.Graph
    :param weight: The edge weight used as the Ricci flow metric. Defaults to "weight".
    :type weight: str
    :param clustering_label: Node attribute name for ground truth.
    :type clustering_label: str
    :param eval_cut: To plot the good guessed cut or not.
    :type eval_cut: bool
    :rtype: A list of floats
    """
    G = G_origin.copy()
    modularity, ari = [], []
    maxw = max(nx.get_edge_attributes(G, weight).values())
    cutoff_range = np.arange(maxw, 1, -0.025)

    best_ari = 0
    best_cutoff = 0

    for cutoff in cutoff_range:
        edge_trim_list = []
        for n1, n2 in G.edges():
            if G[n1][n2][weight] > cutoff:
                edge_trim_list.append((n1, n2))
        G.remove_edges_from(edge_trim_list)

        # Get connected component after cut as clustering
        clustering = {
            c: idx for idx, comp in enumerate(nx.connected_components(G)) for c in comp
        }

        # Compute modularity and ari
        modularity.append(community_louvain.modularity(clustering, G, weight))
        current_ari = ARI(G, clustering, clustering_label=clustering_label)
        ari.append(current_ari)

        # Update best ARI and corresponding cutoff
        if current_ari > best_ari:
            best_ari = current_ari
            best_cutoff = cutoff

    if eval_cut:  # Search for a good cut looking at modularity
        good_cut = -1
        mod_last = modularity[-1]
        drop_threshold = (
            0.01  # at least drop this much to be considered as a drop for good_cut
        )

        # search for drop in modularity
        drops = []
        for i in range(len(modularity) - 1, 0, -1):
            mod_now = modularity[i]
            if mod_last > 1e-4 and (mod_now / mod_last) < drop_threshold:
                drops.append((cutoff_range[i + 1], mod_last))

            mod_last = mod_now

        # Find the tuple with the highest modularity value in drops
        best_drop = max(drops, key=lambda x: x[1])
        # Extract the cutoff value and modularity value
        good_cut = best_drop[0]
        print(
            f"\nBest ARI: {best_ari:.{dcp+1}f}, with cutoff = {best_cutoff:.{dcp}f}\nGuessed cutoff = {good_cut:.{dcp}f}"
        )
        return maxw, cutoff_range, modularity, ari, good_cut

    else:
        print(f"\nBest ARI: {best_ari:.{dcp+1}f}, with cutoff = {best_cutoff:.{dcp}f}")
        return maxw, cutoff_range, modularity, ari
