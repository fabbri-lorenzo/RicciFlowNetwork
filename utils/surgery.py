import community.community_louvain as community_louvain
import networkx as nx
import numpy as np
from sklearn import preprocessing, metrics

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
        print("\nINVALID CLUSTERING TYPE, couldn't compute ARI successfully.\n")
        return -1
    return metrics.adjusted_rand_score(y_true, y_pred)


def perform_surgery(G, weight="weight", clustering_label="community", cut=0):
    """
    A simple surgery function that removes the edges with weight above a threshold.

    :param G: A graph with ``weight`` as the Ricci flow metric to cut.
    :type G: networkx.Graph
    :param weight: The edge weight used as the Ricci flow metric. Defaults to "weight".
    :type weight: str
    :param cut: Manually assigned cutoff point.
    :type cut: float or None
    """
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
    print("* ARI now: %f " % ARI(G, cc, clustering_label))
    print("****************************************")


def check_accuracy(
    G_origin, weight="weight", clustering_label="community", eval_cut=False
):
    """
    Evaluate the clustering quality while cutting edges with a given weight using different thresholds.

    This function iteratively removes edges based on a weight threshold (Ricci flow metric) and evaluates
    the clustering results using modularity and Adjusted Rand Index (ARI). If `eval_cut` is enabled,
    it also estimates a "good" cut threshold by detecting significant drops in modularity.

    :param G_origin: A graph with ``weight`` as the Ricci flow metric used for edge removal.
    :type G_origin: networkx.Graph
    :param weight: The edge weight attribute used as the Ricci flow metric. Defaults to ``"weight"``.
    :type weight: str
    :param clustering_label: Node attribute name for ground truth communities. Defaults to ``"community"``.
    :type clustering_label: str
    :param eval_cut: Whether to compute an estimated optimal cut threshold based on modularity drops. Defaults to ``False``.
    :type eval_cut: bool

    :returns: A tuple containing:
        - **maxw** (*float*): Maximum edge weight value in the graph.
        - **cutoff_range** (*numpy.ndarray*): Array of tested cutoff values for edge removal.
        - **modularity** (*list[float]*): Modularity values at each cutoff.
        - **ari** (*list[float]*): ARI values at each cutoff.
        - (**Optional**) **good_cut** (*float*): Estimated best cutoff based on modularity drop (only if ``eval_cut=True``).
        - (**Optional**) **best_ari** (*float*): Highest ARI value achieved (only if ``eval_cut=True``).
        - (**Optional**) **best_mod** (*float*): Modularity value at best_ari (only if ``eval_cut=True``).

    :rtype: tuple
        - If ``eval_cut=False`` → ``(float, numpy.ndarray, list[float], list[float])``
        - If ``eval_cut=True`` → ``(float, numpy.ndarray, list[float], list[float], float, float, float)``
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
        best_mod = best_drop[1]
        print(
            f"\nBest ARI: {best_ari:.{dcp+1}f}, with cutoff = {best_cutoff:.{dcp}f}\nGuessed cutoff = {good_cut:.{dcp}f}"
        )
        return maxw, cutoff_range, modularity, ari, good_cut, best_ari, best_mod

    else:
        print(f"\nBest ARI: {best_ari:.{dcp+1}f}, with cutoff = {best_cutoff:.{dcp}f}")
        return maxw, cutoff_range, modularity, ari


def get_best_cut(G_origin, weight="weight", clustering_label="value"):
    """
    Determine the best edge removal cutoff threshold to maximize clustering accuracy (ARI).

    This function iteratively removes edges based on a weight threshold (e.g., Ricci flow metric)
    and evaluates the resulting clustering accuracy using the Adjusted Rand Index (ARI).
    The best cutoff is selected as the one that yields the highest ARI.

    :param G_origin: A graph where edges have an attribute ``weight`` used as a removal criterion.
    :type G_origin: networkx.Graph
    :param weight: The edge weight attribute used as the removal metric. Defaults to ``"weight"``.
    :type weight: str
    :param clustering_label: Node attribute name for ground truth communities. Defaults to ``"value"``.
    :type clustering_label: str

    :returns: The cutoff threshold that results in the highest ARI.
    :rtype: float
    """
    G = G_origin.copy()
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

        current_ari = ARI(G, clustering, clustering_label=clustering_label)

        # Update best ARI & best cutoff
        if current_ari > best_ari:
            best_ari = current_ari
            best_cutoff = cutoff

    return best_cutoff
