# Set parameters for graph representations and Ricci Flow
seed_value = 5
iterations_number = 6
rounds = 3
alpha_value = 0.2
surgery_threshold = -1
increment_sr_th_each_round = 0.3


# Parameters for GNet graph
nodes_in_communities = [
    100,
    70,
    50,
    30,
    20,
    20,
    15,
    15,
]  # number of nodes in each community
num_communities = len(nodes_in_communities)


# Community colors
community_colors_data = {
    0: "#712863",  # byzantium
    1: "#0047aa",  # cobalt
    2: "#aae0af",  # celadon
    3: "#ff7f4f",  # coral
    4: "#ffbdd9",  # cotton candy
    5: "#ffa700",  # orange
    6: "#a4c639",  # android green
    7: "#ff355e",  # radical red
    8: "#2e5894",  # sapphire blue
    9: "#dda0dd",  # plum
    10: "#f19cbb",  # pastel pink
    11: "#7f1734",  # claret
    12: "#8f00ff",  # violet
    13: "#4ca3dd",  # light sky blue
}  # Extra colors are present in case algorithm overdetects communities
