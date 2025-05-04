import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

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
    pos = nx.spring_layout(G, seed=42)

    nx.draw(G, pos, with_labels=False, node_color='red', edge_color='gray', node_size=50, font_size=3)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Storm of Swords Graph")
    plt.show()



def plot_log_log_degree_distribution(G, label):
    degrees = [d for n, d in G.degree()]
    degree_counts = np.bincount(degrees)
    nonzero_degrees = np.nonzero(degree_counts)[0]
    probs = degree_counts[nonzero_degrees] / sum(degree_counts)

    plt.plot(nonzero_degrees, probs, marker='o', linestyle='None', label=label)

def plot_log_histogram(G, label):
    degrees = [d for _, d in G.degree()]
    plt.hist(degrees, bins=np.logspace(np.log10(1), np.log10(max(degrees)+1), 15),
             density=True, alpha=0.7, label=label)

def generate_and_compare_PA_graph_with_hist():
    G_real = build_graph()
    n = G_real.number_of_nodes()
    avg_degree = sum(dict(G_real.degree()).values()) / n
    m = max(1, round(avg_degree / 2))
    G_pa = nx.barabasi_albert_graph(n=n, m=m, seed=42)

    plt.figure(figsize=(10, 6))
    plot_log_histogram(G_real, label='AS Graph')
    plot_log_histogram(G_pa, label='Preferential Attachment')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Degree (log)")
    plt.ylabel("PDF (log)")
    plt.title("Degree Distribution (Log-Log Histogram)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return G_pa




# build_graph()
# draw_Graph(build_graph())
generate_and_compare_PA_graph_with_hist()







# def (x):
#     '''
#     f(5 7)
#    x;
#
#     '''
