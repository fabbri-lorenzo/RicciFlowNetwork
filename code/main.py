from GraphRicciCurvature.OllivierRicci import OllivierRicci
from sklearn.metrics import adjusted_rand_score
import networkx as nx
import matplotlib.pyplot as plt


from utils import (
    load_network,
    compute_ricci_flow,
    perform_surgery,
    visualize_communities,
    evaluate_communities,
)


# Load the Facebook Ego Network
graph = nx.read_adjlist("facebook_combined.txt", nodetype=int)

# Compute Ollivier Ricci curvature
orc = OllivierRicci(
    graph, alpha=0.5
)  # alpha controls the weight for local vs global structure
orc.compute_ricci_curvature()

# add the values to the graph edges to track which edges are positively or negatively curved.
for u, v, d in graph.edges(data=True):
    d["ricciCurvature"] = orc.G[u][v]["ricciCurvature"]


# Run the Ricci Flow process
compute_ricci_flow(graph, iterations=15)

# Perform surgery after Ricci flow
perform_surgery(graph, threshold=4)


# Visualize the results
pos = nx.spring_layout(graph)
curvature_values = [d["ricciCurvature"] for u, v, d in graph.edges(data=True)]
nx.draw(
    graph,
    pos,
    edge_color=curvature_values,
    edge_cmap=plt.cm.RdBu,
    node_size=50,
    with_labels=False,
)
plt.show()
