# Standard library imports
import pickle

# Third-party imports
import networkx as nx
import matplotlib.pyplot as plt

# Imports form header files
from graph_drawing import GraphDrawer
from utils import (
    generate_planar_graph_with_communities,
    apply_ricci_flow_and_surgery,
    detect_communities,
    analyze_communities,
    is_converging,
)

# Import from configuration file
import config as cfg


def main():
    # Load the graph from a file
    with open(
        f"PlanarGraphsResults/Graph{cfg.nodes_in_communities}/graph{cfg.nodes_in_communities}.pkl",
        "rb",
    ) as f:
        graph, actual_communities, node_colors = pickle.load(f)

    GraphDrawer(graph, seed=cfg.seed_value).basic_draw(
        title=f"Original Planar Graph {cfg.nodes_in_communities} with Communities",
        node_color=node_colors,
    )

    # Step 3: Apply Ricci flow and edge surgery
    ricci_graph = apply_ricci_flow_and_surgery(
        graph,
        iterations=cfg.iterations_number,
        rounds=cfg.rounds,
        threshold=cfg.surgery_threshold,
        increment_sr_th_each_round=cfg.increment_sr_th_each_round,
        alpha=cfg.alpha_value,
    )
    GraphDrawer(ricci_graph, seed=cfg.seed_value).draw_with_curvatures(
        title=f"Planar Graph {cfg.nodes_in_communities} After Ricci Flow",
        node_color=node_colors,
    )

    if not nx.check_planarity(ricci_graph)[0]:
        print("\nRicci_graph is not planar!!\n")
    elif (
        is_converging(ricci_graph) == True
    ):  # go ahead only if Ricci Flow has converged

        ## Step 4: Detect communities in the updated graph
        pruned_ricci_graph, detected_communities = detect_communities(ricci_graph)

        GraphDrawer(pruned_ricci_graph, seed=cfg.seed_value).basic_draw(
            title=f"Planar Graph {cfg.nodes_in_communities} After Pruning",
            node_color=node_colors,
        )
        # print(f"Detected communities: {detected_communities}")

        ## Step 5: Analyze the communities
        analyze_communities(
            "Community Analysis",
            detected_communities,
            actual_communities,
            community_colors=cfg.community_colors_data,
        )

        # Print global parameters
        print(
            "\n\nIterations: " + str(cfg.iterations_number),
            " Rounds: " + str(cfg.rounds),
            " Alpha: " + str(cfg.alpha_value),
            "Surgery: " + str(cfg.surgery_threshold),
            "Increment to surgery threshold each round: "
            + str(cfg.increment_sr_th_each_round),
            "\n",
        )


def create_and_test():
    # Step 1: Create a planar graph
    graph, actual_communities, node_colors = generate_planar_graph_with_communities(
        cfg.num_communities,
        cfg.nodes_in_communities,
        title=f"graph{cfg.nodes_in_communities}.pkl",
    )

    if not nx.check_planarity(graph)[0]:
        print("\ngraph is not planar!!\n")

    print(f"\nActual Communities: {actual_communities}")
    GraphDrawer(graph, seed=cfg.seed_value).basic_draw(
        title=f"Original Planar Graph with Communities {cfg.nodes_in_communities}",
        node_color=node_colors,
    )


if __name__ == "__main__":
    main()
    # create_and_test()
