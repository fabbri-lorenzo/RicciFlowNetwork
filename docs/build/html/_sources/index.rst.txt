.. RicciFlowNetwork documentation master file, created by
   sphinx-quickstart on Thu Jan 23 14:31:25 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

RicciFlowNetwork's Documentation
============================================
In this project, we apply Ollivier-Ricci curvature and Ricci Flow to detect the two known communities in Zachary’s Karate Club graph. 
In the test modules one can apply the method to two different synthetic graphs to verify its correct behavior. It is also possible to compare the performances of OTD and ATD methods for computing curvature.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Code Workflow
=============
.. mermaid::

   flowchart TB;
    A[(Graph)] --> B[Compute **Ollivier-Ricci Curvature**];
    B --> C[Perform **Ricci Flow**]; 

    subgraph S["Get **Surgery Treshold**"]
        direction LR;
        D@{ shape: diamond, label: "Plot ARI 
        and Modularity"} --> E[Best Cut from ARI]; 
        D@{ shape: diamond, label: "Plot ARI 
        and Modularity"} --> F[Best Cut from Modularity];
    end;

    C --> S;
    S --> U@{ shape: manual-input, label: "User Threshold"};
    U --> G[Perform **Surgery**];
    G --> H[Visualize Detected **Communities**];
    H --> I[Compare with Other Methods];



