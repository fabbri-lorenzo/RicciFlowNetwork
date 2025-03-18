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
import time


def get_exe_time_and_ARI(G, iterations, method):
    """
    Computes execution time and Adjusted Rand Index (ARI) for a given graph using Ricci curvature and flow.

    :param G: The input graph.
    :type G: networkx.Graph
    :param iterations: Number of iterations for Ricci flow computation.
    :type iterations: int
    :param method: The Ricci curvature method to be used ("OTD" or "ATD").
    :type method: str
    :return: A tuple containing execution time (in seconds) and ARI value.
    :rtype: tuple(float, float)
    """
    
    complex_list = nx.get_node_attributes(G, "community")
    for node, value in complex_list.items():
        if isinstance(value, set):
            complex_list[node] = str(value)

    nx.set_node_attributes(G, complex_list, "community")

    start_time = time.time()

    print("Computing Ricci Curvature... ", end="")
    orc = OllivierRicci(G, alpha=0.5, method=method)
    orc.compute_ricci_curvature()

    print("Computing Ricci flow... ", end="")
    orc.compute_ricci_flow(iterations=iterations)

    print("Performing surgery... ")
    best_cut = get_best_cut(orc.G, clustering_label="community")
    perform_surgery(orc.G, cut=best_cut)
    
    cc = nx.connected_components(orc.G)
    ari = ARI(orc.G, list(cc), "community")
    print(f"ARI={ari}")
    
    end_time = time.time()
    exe_time = end_time - start_time

    return exe_time, ari


def plot_exe_times_and_ARI(data_OTD, data_ATD, nodes, x_values):
    """
    Plots execution times and ARI values for OTD and ATD methods.

    :param data_OTD: Execution times and ARIs for the OTD method.
    :type data_OTD: list of tuple(float, float)
    :param data_ATD: Execution times and ARIs for the ATD method.
    :type data_ATD: list of tuple(float, float)
    :param nodes: Number of nodes in the graph.
    :type nodes: int
    :param x_values: X-axis values representing iteration numbers.
    :type x_values: numpy array
    """
    
    times_OTD = [res[0] for res in data_OTD]  
    ARIs_OTD = [res[1] for res in data_OTD]  
    
    times_ATD = [res[0] for res in data_ATD]  
    ARIs_ATD = [res[1] for res in data_ATD] 
    
    # Creating the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 1]})
    
    # -- TOP subplot: ARIs for OTD/ATD --
    ax1.plot(x_values, ARIs_OTD, marker='o', linestyle='None', color='midnightblue', label="ARI OTD")
    ax1.plot(x_values, ARIs_ATD, marker='o', linestyle='None', color='firebrick', label="ARI ATD")
    ax1.set_ylabel("ARI")
    ax1.legend()
    
    # -- BOTTOM subplot: execution times as histograms --
    width = 0.5
    ax2.bar(x_values - width/2, times_OTD, width=width, color='midnightblue', label="Times OTD")
    ax2.bar(x_values + width/2, times_ATD, width=width, color='firebrick', label="Times ATD")
    ax2.set_ylabel("Execution Time (s)")
    ax2.legend()
    
    plt.xlabel("Iterations")
    fig.suptitle(f"Ricci Flow on LFR graph with {nodes} nodes, $\\mu$=0.20")
    
    save_path = "tests/LFRResults"
    plt.tight_layout()
    plt.gcf().canvas.manager.set_window_title("LFR Execution Times Graph")
    plt.savefig(os.path.join(save_path, f"Exe_times_LFR_{nodes}_nodes.png"), dpi=600)
    plt.show()
    

def exe_times_LFR():
    """
    Generates an LFR benchmark graph and evaluates execution time and ARI for OTD and ATD methods.

    :return: None
    """
    
    nodes = 2500
    iterations_list = np.arange(5, 35, 5)
    
    # Parameters for different node sizes
    min_com = {500: 30, 1000: 20, 2500: 70}
    max_com = {500: 50, 1000: 40, 2500: 130}
    max_deg = {500: 60, 1000: 45, 2500: 150}
    
    G = nx.LFR_benchmark_graph(
        n=nodes,  
        tau1=2.5,  
        tau2=1.8,  
        mu=0.20,  
        min_community=min_com.get(nodes),  
        max_community=max_com.get(nodes),  
        average_degree=20,  
        max_degree=max_deg.get(nodes),  
        max_iters=1000,  
        seed=42,  
    )
    
    print("Graph generated. ", end="\n")
    
    # Compute execution times and ARIs for OTD and ATD
    data_OTD = [
        get_exe_time_and_ARI(G, iterations, "OTD")
        for iterations in tqdm(iterations_list, desc=f"\nProcessing graph with {nodes} nodes, OTD")
    ]
    data_ATD = [
        get_exe_time_and_ARI(G, iterations, "ATD")
        for iterations in tqdm(iterations_list, desc=f"\nProcessing graph with {nodes} nodes, ATD")
    ]
    
    plot_exe_times_and_ARI(data_OTD, data_ATD, nodes, iterations_list)


if __name__ == "__main__":
    exe_times_LFR()
