# from matplotlib import gridspec
import networkx as nx
import random
import matplotlib.pyplot as plt


# Parameters for the random graph
# num_nodes = 120
# edge_prob = 0.07

# Create a random graph
# G = nx.erdos_renyi_graph(num_nodes, edge_prob)

# Load your graph
facebookGraph = nx.read_edgelist("facebook_combined.txt")

sampled_nodes = random.sample(list(facebookGraph.nodes()), 150)  # Convert dict_keys to list
G = facebookGraph.subgraph(sampled_nodes)

degree_dict = dict(G.degree())

# Set Adam to be the node with the highest degree
adam = max(degree_dict, key=degree_dict.get)

# Get Adam's neighbors and create the subgraph of Adam's neighborhood
adam_neighbors = list(G.neighbors(adam))
adam_neighborhood_nodes = [adam] + adam_neighbors
adam_component = G.subgraph(adam_neighborhood_nodes).copy()  # Create subgraph with Adam and his neighbors

# Adam statistics
print(f"Nodes: {adam_component.number_of_nodes()}")
print(f"Edges: {adam_component.number_of_edges()}")

connected_components = list(nx.connected_components(adam_component))
print(f"Connected components: {len(connected_components)}")

print(f"Size of the largest connected component: {len(max(connected_components, key=len))}")


# Degree distribution for Adam's neighborhood
degree_sequence = list(dict(adam_component.degree()).values())  # Get the degrees of all nodes in Adam's component

# Create a dictionary to count the frequency of each degree in Adam's neighborhood
degree_count_adam = {}
for degree in degree_sequence:
    degree_count_adam[degree] = degree_count_adam.get(degree, 0) + 1

print("Degree Distribution for Adam's Component:", degree_count_adam)




# Calculate the number of triangles that include Adam
triangle_count = 0
for i in range(len(adam_neighbors)):
    for j in range(i + 1, len(adam_neighbors)):
        if adam_component.has_edge(adam_neighbors[i], adam_neighbors[j]):
            triangle_count += 1

print(f"Number of triangles including Adam: {triangle_count}")
# C(N,2)
# max triangle amount
num_neighbors = len(list(adam_component.neighbors(adam)))

# The maximum number of triangles Adam could form is the number of ways to pick 2 neighbors from n neighbors
max_triangles = (num_neighbors * (num_neighbors - 1)) // 2
# TODO

print(f"Maximum number of triangles Adam's network could have: {max_triangles}")



# Adam closest friend
best_friend = None
max_shared_neighbors = -1

for neighbor in adam_neighbors:

    neighbor_neighbors = set(G.neighbors(neighbor))

    shared_neighbors = set(adam_neighbors).intersection(neighbor_neighbors)

    if len(shared_neighbors) > max_shared_neighbors:
        max_shared_neighbors = len(shared_neighbors)
        best_friend = neighbor

print(f"Adam's best friend is: {best_friend} with {max_shared_neighbors} shared neighbors.")

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

print(f"Small Shared Subgraph - Nodes: {subgraph_small_shared.number_of_nodes()} Edges: {subgraph_small_shared.number_of_edges()}")
print(f"Large Shared Subgraph - Nodes: {subgraph_large_shared.number_of_nodes()} Edges: {subgraph_large_shared.number_of_edges()}")



# graph representation

plt.figure(figsize=(8, 6))
plt.title("Original Graph")
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=False, node_size=30, edge_color="gray")
plt.show()

plt.figure(figsize=(8, 6))
plt.title("adam Graph")
pos = nx.spring_layout(adam_component)
nx.draw(adam_component, pos, with_labels=False, node_size=30, edge_color="gray")
plt.show()

plt.figure(figsize=(8, 6))
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
nx.draw_networkx_nodes(adam_component, pos, nodelist=[best_friend], node_size=100, node_color="blue", edgecolors="black")
plt.show()


degree_sequence = list(dict(adam_component.degree()).values())
plt.figure(figsize=(8, 6))
plt.hist(degree_sequence, bins=range(min(degree_sequence), max(degree_sequence) + 2), edgecolor="black", align='left')
plt.title("Degree Distribution")
plt.xlabel("Degree")
plt.ylabel("Number of Nodes")
plt.show()








#
# fig = plt.figure(figsize=(9, 3))
# gs = gridspec.GridSpec(2, 3, height_ratios=[1, 0.3], width_ratios=[1, 1, 0.5])  # 2 rows, 3 columns
#
# # Plot the original graph in the first grid cell (top-left)
# ax1 = plt.subplot(gs[0, 0])
# ax1.set_title("Original Graph")
# pos = nx.spring_layout(G)  # Recompute position for the graph layout
# nx.draw(G, pos, with_labels=False, node_size=30, edge_color="gray", ax=ax1)
# plt.show()
#
# # Plot Adam's component in the second grid cell (top-middle)
# ax2 = plt.subplot(gs[0, 1])
# ax2.set_title("Adam's Component")
#
# node_colors = []
# for node in adam_component.nodes():
#     if node == adam:
#         node_colors.append('purple')  # Paint Adam in purple
#     elif node in subgraph_small_shared.nodes():
#         node_colors.append('orange')  # Nodes in the small shared subgraph
#     else:
#         node_colors.append('blue')  # Nodes in the large shared subgraph
#
#
# # Draw the updated graph
# nx.draw(adam_component, pos, with_labels=False, node_size=50, node_color=node_colors, edge_color="red", ax=ax2)
# # Highlight Adam's best friend
# nx.draw_networkx_nodes(adam_component, pos, nodelist=[best_friend], node_size=100, node_color="yellow", edgecolors="black", ax=ax2)
# plt.show()



# Plot the degree distribution in the third grid cell (top-right)
# ax3 = plt.subplot(gs[0, 2])
# ax3.hist(degree_sequence, bins=range(min(degree_sequence), max(degree_sequence) + 2), edgecolor="black", align='left')
# ax3.set_title("Degree Distribution")
# ax3.set_xlabel("Degree")
# ax3.set_ylabel("Number of Nodes")
#
# plt.tight_layout()
# plt.gcf().canvas.manager.set_window_title('Adam\'s Network')
# plt.show()
#
#
#
#
#
