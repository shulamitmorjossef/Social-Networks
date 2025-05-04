import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


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

def draw_Graph(G, title="Graph"):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=False, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title(title)
    plt.show()

# --- b.  G(n, m) ---
def build_gnm_graph():
    original = build_graph()
    n = original.number_of_nodes()
    m = original.number_of_edges()
    G = nx.gnm_random_graph(n, m, seed=42)
    return G

# --- c. Configuration model ---
def build_configuration_model():
    original = build_graph()
    degree_seq = [d for n, d in original.degree()]
    G = nx.configuration_model(degree_seq, seed=42)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


draw_Graph(build_graph())
draw_Graph(build_gnm_graph(), "G(n, m) Graph")
draw_Graph(build_configuration_model(), "Configuration Model")
