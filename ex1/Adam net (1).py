import networkx as nx
import random
import matplotlib.pyplot as plt
from math import comb
import numpy as np
from collections import Counter


# Load graph
facebookGraph = nx.read_edgelist("facebook_combined.txt")

sampled_nodes = random.sample(list(facebookGraph.nodes()), 120)
G = facebookGraph.subgraph(sampled_nodes)

degree_dict = dict(G.degree())

# Set Adam to be the node with the highest degree
adam = max(degree_dict, key=degree_dict.get)

# Get Adam's neighbors and create the subgraph of Adam's neighborhood
adam_neighbors = list(G.neighbors(adam))
adam_neighborhood_nodes = [adam] + adam_neighbors
adam_component = G.subgraph(adam_neighborhood_nodes).copy()  # Create subgraph with Adam and his neighbors

# Adam statistics
print(f"-Nodes: {adam_component.number_of_nodes()}")
print(f"-Edges: {adam_component.number_of_edges()}")

num_neighbors = len(list(adam_component.neighbors(adam)))

connected_components = list(nx.connected_components(adam_component))
print(f"-Connected components: {len(connected_components)}")

print(f"-Size of the largest connected component: {len(max(connected_components, key=len))}")


# Degree distribution for Adam's neighborhood
degree_sequence = list(dict(adam_component.degree()).values())  # Get the degrees of all nodes in Adam's component

# Create a dictionary to count the frequency of each degree in Adam's neighborhood
degree_count_adam = {}
for degree in degree_sequence:
    degree_count_adam[degree] = degree_count_adam.get(degree, 0) + 1

print("-Degree Distribution for Adam's Component:", degree_count_adam)




# Calculate the number of triangles that include Adam
triangle_count = 0
for i in range(len(adam_neighbors)):
    for j in range(i + 1, len(adam_neighbors)):
        if adam_component.has_edge(adam_neighbors[i], adam_neighbors[j]):
            triangle_count += 1

print(f"-Number of triangles including Adam: {triangle_count}")

# C(N,2)
# The maximum number of triangles Adam could form is the number of ways to pick 2 neighbors from n neighbors
max_triangles = (num_neighbors * (num_neighbors - 1)) // 2

# max_triangles1 = comb(num_neighbors, 2)

print(f"-Maximum number of triangles Adam's network could have: {max_triangles}")


# Adam closest friend
best_friend = None
max_shared_neighbors = -1

for neighbor in adam_neighbors:

    neighbor_neighbors = set(G.neighbors(neighbor))

    shared_neighbors = set(adam_neighbors).intersection(neighbor_neighbors)

    if len(shared_neighbors) > max_shared_neighbors:
        max_shared_neighbors = len(shared_neighbors)
        best_friend = neighbor

print(f"-Adam's best friend (base on the largest common neighbors) is: {best_friend} with {max_shared_neighbors} shared neighbors.")

# dividing to subgroups
subgraph_small_shared = []
subgraph_large_shared = []

for neighbor in adam_neighbors:
    neighbor_neighbors = set(G.neighbors(neighbor))

    shared_neighbors = set(adam_neighbors).intersection(neighbor_neighbors)

    if len(shared_neighbors) < 2:
        subgraph_small_shared.append(neighbor)
    else:
        subgraph_large_shared.append(neighbor)


subgraph_small_shared = G.subgraph([adam] + subgraph_small_shared).copy()
subgraph_large_shared = G.subgraph([adam] + subgraph_large_shared).copy()

node_color = []
for node in subgraph_small_shared.nodes():
    node_color.append('orange')

for node in subgraph_large_shared.nodes():
    node_color.append('blue')

print(f"-Small Shared Subgraph - Nodes: {subgraph_small_shared.number_of_nodes()} Edges: {subgraph_small_shared.number_of_edges()}")
print(f"-Large Shared Subgraph - Nodes: {subgraph_large_shared.number_of_nodes()} Edges: {subgraph_large_shared.number_of_edges()}")


# graph representation
plt.figure(figsize=(8, 4))
plt.title("Original Graph")
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=False, node_size=30, edge_color="gray")
plt.show()

plt.figure(figsize=(8, 4))
plt.title("adam Graph")
pos = nx.spring_layout(adam_component)
nx.draw(adam_component, pos, with_labels=False, node_size=30, edge_color="gray")
plt.show()

plt.figure(figsize=(8, 4))
plt.title("Adam's Neighborhood")

node_colors = []
for node in adam_component.nodes():
    if node == adam:
        node_colors.append('purple')
    elif node in subgraph_small_shared.nodes():
        node_colors.append('orange')
    else:
        node_colors.append('blue')

nx.draw(adam_component, pos, with_labels=False, node_size=50, node_color=node_colors, edge_color="red")
nx.draw_networkx_nodes(adam_component, pos, nodelist=[best_friend], node_size=100, node_color="yellow", edgecolors="black")
plt.show()


degree_sequence = list(dict(adam_component.degree()).values())
plt.figure(figsize=(8, 4))
# avg = sum(degree_sequence) / len(degree_sequence)
# newS = list(map(lambda x: x/avg, degree_sequence))

plt.hist(degree_sequence, bins=range(min(degree_sequence), max(degree_sequence) + 2), edgecolor="black", align='left')

plt.title("Degree Distribution")
plt.xlabel("Degree")
plt.ylabel("Number of Nodes")
plt.show()
#
degrees = np.array([deg for _, deg in adam_component.degree()])

# נרמול הדרגות לטווח [0,1] (שיטת מינ-מקס)
min_d, max_d = degrees.min(), degrees.max()
normalized_degrees = (degrees - min_d) / (max_d - min_d) if max_d != min_d else np.ones_like(degrees)

# חישוב התפלגות הדרגות המנורמלות
degree_counts = Counter(normalized_degrees)
x = sorted(degree_counts.keys())
y = [degree_counts[d] / len(adam_component.nodes()) for d in x]  # נרמול לפי כמות הצמתים

# ציור ההיסטוגרמה
plt.figure(figsize=(8, 6))
plt.bar(x, y, color="blue", edgecolor="black", alpha=0.7, width=0.05)
plt.xlabel("Normalized Degree")
plt.ylabel("Probability")
plt.title("Normalized Degree Distribution in Adam's Neighborhood")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()

# Degree Distribution (Normalized)
# total_nodes = len(a.nodes)
# degrees = [degree for node, degree in adam_ego_graph.degree()]
# degree_counts = Counter(degrees)
#
# x = sorted(degree_counts.keys())
# y = [degree_counts[d] / total_nodes for d in x] #nirmul 1
#
# plt.figure(figsize=(8, 6))
# plt.bar(x, y, color="blue", edgecolor="black", alpha=0.7)
# plt.xlabel("Degree")
# plt.ylabel("Probability")
# plt.title("Normalized Degree Distribution")
# plt.grid(axis="y", linestyle="--", alpha=0.5)
# plt.show()
