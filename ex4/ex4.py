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


# build_graph()
draw_Graph(build_graph())