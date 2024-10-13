# Community Detection on Facebook Ego Network with Ricci Flow

This repository implements community detection on the **Facebook Ego Network** using **Discrete Ricci Flow**, based on the methodology presented in the paper ["Community Detection on Networks with Ricci Flow"](https://doi.org/10.1038/s41598-019-46380-9) by Chien-Chun Ni et al.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
  - [1. Load the Network](#1-load-the-network)
  - [2. Compute Ollivier-Ricci Curvature](#2-compute-ollivier-ricci-curvature)
  - [3. Apply Ricci Flow](#3-apply-ricci-flow)
  - [4. Perform Network Surgery](#4-perform-network-surgery)
  - [5. Visualize Communities](#5-visualize-communities)
  - [6. Evaluate Results](#6-evaluate-results)
- [Results](#results)
- [References](#references)

## Introduction
This project uses **Ricci Flow** as a geometric approach to detect community structures in networks. It applies **Ollivier-Ricci curvature** to adjust the weights of edges in a graph, iterating the process to shrink intra-community edges and stretch inter-community edges. The **Facebook Ego Network** is used as the dataset, with the goal of identifying the known friend circles (communities) using the Ricci Flow method.

## Requirements
The following Python libraries are required:
- `networkx`
- `GraphRicciCurvature`
- `matplotlib`
- `numpy`
- `scikit-learn` (for evaluation metrics)

To install these, run:
```bash
pip3 install networkx ricci-graph-tool numpy matplotlib scikit-learn
```


## Data
The Facebook Ego Network data is used in this project. You can download the dataset from the [SNAP dataset collection](http://snap.stanford.edu/data/egonets-Facebook.html).

## Usage

### 1. Load the Network
To load the Facebook Ego Network into Python:
```python
import networkx as nx

# Load the Facebook Ego Network
graph = nx.read_adjlist("facebook_combined.txt", nodetype=int)
```

### 2. Compute Ollivier-Ricci Curvature
You can compute the Ollivier-Ricci curvature for each edge in the network using:

```python
from GraphRicciCurvature.OllivierRicci import OllivierRicci

orc = OllivierRicci(graph, alpha=0.5)  # Adjust alpha for curvature sensitivity
orc.compute_ricci_curvature()

# Add curvature to edge attributes
for u, v, d in graph.edges(data=True):
    d['ricciCurvature'] = orc.G[u][v]['ricciCurvature']
```

### 3. Apply Ricci Flow
Simulate the discrete Ricci Flow process by iteratively adjusting edge weights based on their curvature:

```python
def ricci_flow(graph, iterations=10):
    for i in range(iterations):
        for u, v, d in graph.edges(data=True):
            curvature = d['ricciCurvature']
            weight_update = curvature * d.get('weight', 1)
            d['weight'] = max(d.get('weight', 1) + weight_update, 0.1)

# Run Ricci Flow for 15 iterations
ricci_flow(graph, iterations=15)
```


### 4. Perform Network Surgery
Remove high-weight inter-community edges to detect communities:

```python
def perform_surgery(graph, threshold=4):
    edges_to_remove = [(u, v) for u, v, d in graph.edges(data=True) if d['weight'] > threshold]
    graph.remove_edges_from(edges_to_remove)

# Perform surgery after Ricci Flow
perform_surgery(graph, threshold=4)
```

### 5. Visualize Communities
Visualize the network with color-coded communities:

```python
import matplotlib.pyplot as plt

pos = nx.spring_layout(graph)
curvature_values = [d['ricciCurvature'] for u, v, d in graph.edges(data=True)]
nx.draw(graph, pos, edge_color=curvature_values, edge_cmap=plt.cm.RdBu, node_size=50, with_labels=False)
plt.show()
```

### 6. Evaluate results
Compare the detected communities with ground-truth friend circles using Adjusted Rand Index (ARI):

```python
from sklearn.metrics import adjusted_rand_score

# Ground-truth and detected labels (for evaluation purposes)
true_labels = [...]  # Ground-truth communities
detected_labels = [...]  # Detected from the algorithm

ari = adjusted_rand_score(true_labels, detected_labels)
print(f"Adjusted Rand Index: {ari}")
```

## Results
After running the Ricci Flow algorithm on the Facebook Ego Network, communities can be identified based on the edge curvatures. The resulting network will reveal clearly separated clusters corresponding to the original friend circles. Further evaluation metrics like ARI and modularity can be used to validate the accuracy of the detected communities.

## References
1. Ni, C.-C., Lin, Y.-Y., Luo, F., & Gao, J. (2019). Community Detection on Networks with Ricci Flow. *Scientific Reports*, 9:9984. [https://doi.org/10.1038/s41598-019-46380-9](https://doi.org/10.1038/s41598-019-46380-9)
2. SNAP Dataset Collection: [http://snap.stanford.edu/data/egonets-Facebook.html](http://snap.stanford.edu/data/egonets-Facebook.html)
