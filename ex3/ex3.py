import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random

from scipy.stats import linregress

file = pd.read_csv("stormofswords.csv").iloc[1:]
tribes = pd.read_csv("tribes.csv")

nodes = tribes.iloc[:, 0].tolist()
edges = list(zip(file["Source"], file["Target"], file["Weight"]))



def build_graph():
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for src, tgt, weight in edges:
        G.add_edge(src, tgt, weight=weight)
    return G

def draw_Graph(G):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)  # פריסה אקראית אך קבועה

    # ציור הקשתות עם משקלים
    nx.draw(G, pos, with_labels=False, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Storm of Swords Graph")
    plt.show()

def giant_component_sizes(G, edge_order):
    sizes = []
    for edge in edge_order:
        if G.has_edge(*edge):
            G.remove_edge(*edge)
        if len(G.edges) > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            sizes.append(len(largest_cc))
        else:
            sizes.append(0)
    return sizes

def create_orders_and_draw():
    G_random = build_graph()
    G_heavy_first = build_graph()
    G_light_first = build_graph()
    G_betweenness = build_graph()

    # 1. רנדומלי
    random_edges = list(G_random.edges())
    random.shuffle(random_edges)
    sizes_random = giant_component_sizes(G_random, random_edges)

    # 2. כבדות -> קלות
    heavy_edges = sorted(G_heavy_first.edges(data=True), key=lambda x: -x[2]['weight'])
    heavy_edges_list = [(u, v) for u, v, _ in heavy_edges]
    sizes_heavy = giant_component_sizes(G_heavy_first, heavy_edges_list)

    # 3. קלות -> כבדות
    light_edges = sorted(G_light_first.edges(data=True), key=lambda x: x[2]['weight'])
    light_edges_list = [(u, v) for u, v, _ in light_edges]
    sizes_light = giant_component_sizes(G_light_first, light_edges_list)

    # 4. לפי Betweenness
    betweenness = nx.edge_betweenness_centrality(G_betweenness)
    betweenness_edges = sorted(betweenness.items(), key=lambda x: -x[1])
    betweenness_edges_list = [edge for edge, _ in betweenness_edges]
    sizes_betweenness = giant_component_sizes(G_betweenness, betweenness_edges_list)

    draw_Graph(G)

    # ציור גרף אחד עם ארבע עקומות
    plt.figure(figsize=(12, 8))
    plt.plot(sizes_random, label="Random Removal", color='blue')
    plt.plot(sizes_heavy, label="Heavy → Light", color='red')
    plt.plot(sizes_light, label="Light → Heavy", color='green')
    plt.plot(sizes_betweenness, label="High Betweenness", color='purple')

    plt.xlabel("Edges Removed")
    plt.ylabel("Size of Giant Component")
    plt.title("Edge Removal and Giant Component Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    axs[0].plot(sizes_random, color='blue')
    axs[0].set_title("Random Edge Removal")

    axs[1].plot(sizes_heavy, color='red')
    axs[1].set_title("Heavy to Light Edge Removal")

    axs[2].plot(sizes_light, color='green')
    axs[2].set_title("Light to Heavy Edge Removal")

    axs[3].plot(sizes_betweenness, color='purple')
    axs[3].set_title("High Betweenness Edge Removal")

    for ax in axs:
        ax.set_xlabel("Edges Removed")
        ax.set_ylabel("Size of Giant Component")
        ax.grid(True)

    plt.tight_layout()
    plt.show()


# Function to calculate the neighborhood overlap for each edge in the graph
def calculate_neighborhood_overlap(G):
    overlaps = []  # List to store overlap coefficients
    weights = []  # List to store edge weights

    for u, v, data in G.edges(data=True):  # Iterate through all edges in the graph
        neighbors_u = set(G.neighbors(u))  # Get the neighbors of node u
        neighbors_v = set(G.neighbors(v))  # Get the neighbors of node v

        common = neighbors_u & neighbors_v  # Find the common neighbors between u and v
        union = neighbors_u | neighbors_v  # Find the union of neighbors between u and v

        # Calculate the overlap coefficient (common neighbors / total unique neighbors)
        overlap = len(common) / len(union) if len(union) > 0 else 0

        overlaps.append(overlap)  # Append the overlap coefficient to the list
        weights.append(data['weight'])  # Append the weight of the edge to the list

    return overlaps, weights  # Return the overlap coefficients and weights

# # Function to plot the neighborhood overlap vs. edge weight with a trend line
def plot_neighborhood_overlap(G, title, filename):
    overlaps, weights = calculate_neighborhood_overlap(G)  # Get overlap and weight data

    # Calculate the linear regression trend line (slope and intercept)
    slope, intercept, _, _, _ = linregress(weights, overlaps)
    trend_y = [slope * w + intercept for w in weights]  # Calculate the trend line values

    # Create the plot with white background
    plt.style.use('default')  # Use default (white) style
    plt.figure(figsize=(10, 6), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')  # Set axis background to white

    # Scatter plot of data points with colorful points
    plt.scatter(weights, overlaps, alpha=0.7, color='blue', edgecolors='green', linewidths=0.3, label='Edges')

    # Plot the trend line with red color
    plt.plot(weights, trend_y, linestyle='--', color='red', linewidth=2, label='Trend Line')

    # Set titles and labels with black color (since background is white)
    plt.title(title, fontsize=14, weight='bold', color='black')
    plt.xlabel("Weight", fontsize=12, color='black')
    plt.ylabel("Overlap", fontsize=12, color='black')

    # Set the ticks color to black for contrast
    plt.xticks(color='black')
    plt.yticks(color='black')

    # Enable grid with dashed lines and light alpha
    plt.grid(True, linestyle='--', alpha=0.6)

    # Display legend
    plt.legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the plot to a file with high resolution
    plt.savefig(filename, dpi=300)

    # Show the plot
    plt.show()

G = build_graph()

# create_orders_and_draw()

plot_neighborhood_overlap(G, "Overlap and Weight", "neighborhood_overlap.png")

