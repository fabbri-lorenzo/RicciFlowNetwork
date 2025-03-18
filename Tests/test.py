import networkx as nx

def test(nodes):
# dicts with average degrees as keys and graph parameters as values
    min_com = {500: 30, 1000: 20, 2500: 70, 5000: 100}
    max_com = {500: 50, 1000: 40, 2500: 130, 5000: 160}
    max_deg = {500: 60, 1000: 45, 2500: 150, 5000: 180}
    
    G = nx.LFR_benchmark_graph(
        n=nodes,  # Number of nodes
        tau1=2.5,  # Degree distribution exponent
        tau2=1.8,  # Community size distribution exponent
        mu=0.27,  # Mixing parameter
        min_community=min_com.get(nodes),  # Min num of nodes in each community
        max_community=max_com.get(nodes),  # Max num of nodes in each community
        average_degree=20,  # Average degree per node
        max_degree=max_deg.get(nodes),  # Maximum degree per node
        max_iters=1000,  # Maximum number of iterations allowed to generate the graph
        seed=42,  # Random seed for reproducibility
    )
    
    print("generated. ", end="")
    
test(5000)    