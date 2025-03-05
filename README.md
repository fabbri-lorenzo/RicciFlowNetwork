# Community Detection on Networks using Ricci Flow
This repository implements community detection on **Zachary's Karate Club graph** using **Discrete Ricci Flow** with Python, based on the methodology presented in the paper ["Community Detection on Networks with Ricci Flow"](https://doi.org/10.1038/s41598-019-46380-9) by Chien-Chun Ni et al.

This is a project for the course on Complex Networks of the University of Bologna.
The essay for the exam can be found [here](latex/DiscreteRicciFlow.pdf)

## Table of Contents
  - [Introduction](#introduction)
  - [Requirements](#requirements)
  - [Data](#data)
  - [Usage](#usage)
  - [Results](#results)
  - [References](#references)

## Introduction
In this project, we apply **Ollivier-Ricci curvature** and **Ricci Flow** to detect the two known communities in Zachary’s Karate Club graph. The approach follows the work of Ni et al. (2019), where Ricci Flow is used to reshape edge weights iteratively, enhancing the separation between different communities. After applying Ricci Flow, we perform edge surgery to remove weakly connected edges and extract communities as the connected components of the resulting graph.

**Documentation** can be built inside 'docs' folder using Sphynx, or it is simply accessible [here](https://fancy-dodol-4d2c6d.netlify.app/).

## Requirements
To run this project, you need the following dependencies:

- NetworkX
- NumPy
- Matplotlib
- scikit-learn (for clustering)
- tqdm (for progress bars)

You can install them with:

```bash
pip install networkx numpy matplotlib scikit-learn tqdm
```

## Data
All the datasets employed are loaded using NetworkX. E.g.:

```python
import networkx as nx

G = nx.karate_club_graph()
```


## Usage
The main script performs the following steps:

1. **Load the Karate Club graph** using NetworkX.
2. **Compute the Ollivier-Ricci curvature** for all edges.
3. **Evolve edge weights** using Ricci Flow based on the curvature.
4. **Plots ARI and Modularity vs. Cutoff** and prints on terminal cutoff value corresponding to highest ARI.
5. **(Optional) Guess a good cut based on drop of modularity** (useful in cases where ARI is not available).
6. **Perform edge surgery** by removing weakly connected edges.
7. **Detect communities** using clustering algorithms.
8. **Visualize the results**, including the community structure and the original vs. post-surgery graphs.

### Running the script
Simply execute the following command to run the main script:

```bash
python karate_club.py
```

Ensure that you have the required dependencies installed (see the **Requirements** section for the installation command).

You can test the code on a toymodel with a Stochastic Block Model (SBM) graph and a Lancichinetti-Fortunato-Radicchi (LFR) graph. 
You can also see the difference in performances between Optimal Transportation Method (OTD) and Average Transportation Method (ATD) applied to curvature evaluation in the Ricci Flow process of a LFR graph.
**Before running** either 
```bash
python tests/toymodel.py
```
or 
```bash
python tests/LFR.py
```

you should **change** 

```toymodel/LFR.py
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.getcwd(), "/Users/lorenzofabbri/Downloads/Code/RicciFlowNetwork"
        )  # Substitute with your own path
    )
)
```
with your path to the directory containing the code.

## Results
After running the karate club algorithm, the detected communities are visualized to show how Ricci Flow enhances community separation. The following outputs are generated:

1. **Original Graph and Histograms Visualization**: The initial Karate Club graph with its nodes and edges.
2. **Post-Ricci Flow Graph and Histograms Visualization**: The graph after Ricci Flow has been applied to evolve edge weights and improve community structure.
3. **Behaviour of ARI and Modularity depending on cutoff treshold for surgery**: Plots of ARI and modularity of the graph after hypotetical surgery with different cutoff parameters.
4. **Edge Surgery Visualization**: The graph after weak edges are removed based on a threshold chosen by the user (use the one corresponding to highest ARI for better results).
5. **Community Detection Output**: The nodes are color-coded according to the communities detected after edge surgery.

## References
1. Ni, C.-C., Lin, Y.-Y., Luo, F., & Gao, J. (2019). Community Detection on Networks with Ricci Flow. *Scientific Reports*, 9:9984. [https://doi.org/10.1038/s41598-019-46380-9](https://doi.org/10.1038/s41598-019-46380-9)
2. Wayne W. Zachary (1997). An Information Flow Model for Conflict and Fission in Small Groups. *The University of Chicago Press Journals*, 33:4 [https://doi.org/10.1086/jar.33.4.3629752](https://doi.org/10.1086/jar.33.4.3629752)
