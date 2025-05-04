from collections import Counter
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

from scipy.stats import linregress

# ----- data of network -----
file = pd.read_csv("stormofswords.csv").iloc[1:]
tribes = pd.read_csv("tribes.csv")

nodes = tribes.iloc[:, 0].tolist()
edges = list(zip(file["Source"], file["Target"], file["Weight"]))

# ----- build original graph -----
def build_graph():
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for src, tgt, weight in edges:
        G.add_edge(src, tgt, weight=weight)
    return G


# ----- draw graph -----
def draw_Graph(G, title="Game of Thrones Graph"):
    plt.figure(figsize=(6, 4))

    pos = nx.spring_layout(G, seed=42)

    nx.draw(G, pos, with_labels=False, node_color='red', edge_color='gray', node_size=80, width=1)

    plt.suptitle(title, fontsize=16, fontweight='bold')

    plt.show()

# --- b.  G(n, m) ---
def build_gnm_graph():
    original = build_graph()
    n = original.number_of_nodes()
    m = original.number_of_edges()
    G = nx.gnm_random_graph(n, m)
    return G


# --- c. Configuration model ---
def build_configuration_model():
    original = build_graph()
    degree_seq = [d for n, d in original.degree()]
    G = nx.configuration_model(degree_seq)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

# ------ block model------
def block_model():
    original = build_graph()
    n = original.number_of_nodes()

    # Divide the nodes equally into 3 groups (communities)
    sizes = [n // 3] * 3

    p_in = 0.3
    p_out = 0.02

    # Create the probability matrix for the stochastic block model
    # Diagonal values (i == j) use p_in (within group),
    # Off-diagonal values use p_out (between groups)
    probs = [[p_in if i == j else p_out for j in range(3)] for i in range(3)]

    G_block = nx.stochastic_block_model(sizes, probs)

    return G_block

def estimate_p_from_degrees(G):

    n = G.number_of_nodes()
    if n <= 1:
        return 0

    degrees = [deg for _, deg in G.degree()]
    avg_degree = sum(degrees) / n

    p = avg_degree / (n - 1)
    return p

def build_gnp_graph():
    original = build_graph()
    n = original.number_of_nodes()
    p = estimate_p_from_degrees(original)
    G = nx.erdos_renyi_graph(n=n, p=p)
    return G

def build_preferential_attachment_graph():
    original = build_graph()
    n = original.number_of_nodes()
    m = 3

    G = nx.barabasi_albert_graph(n=n, m=m)
    return G

def plot_degree_distribution(G, title="Degree Distribution"):
    """
    Plot the degree distribution of a graph.

    Parameters:
    - G: networkx graph
    - title: string, title for the plot
    """
    degrees = [d for _, d in G.degree()]
    plt.figure(figsize=(7, 5))
    plt.hist(degrees, bins=range(1, max(degrees) + 2), alpha=0.6, color="red", edgecolor="black")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_all_degree_distribution():
    G = build_graph()
    gnm = build_gnm_graph()
    c = build_configuration_model()
    b = block_model()
    gnp = build_gnp_graph()
    p = build_preferential_attachment_graph()

    all_graphs = {
        "Game of Thrones": G,
        "G(n,p)": gnp,
        "G(n,m)": gnm,
        "Configuration Model": c,
        "Block Model": b,
        "Preferential Attachment": p
    }

    for name, G in all_graphs.items():
        plot_degree_distribution(G, name)

def draw_all_graphes():
    draw_Graph(build_graph())
    draw_Graph(build_gnm_graph(), "G(n, m) Graph")
    draw_Graph(build_configuration_model(), "Configuration Model Graph")
    draw_Graph(block_model(), "Block Model Graph")
    draw_Graph(build_gnp_graph(), "G(n, p) Graph")
    draw_Graph(build_preferential_attachment_graph(), "preferential Attachment Graph")

def giant_component():
    """
    Plot the Giant Component of the graph.

    Parameters:
    - G: networkx graph
    """
    G = build_graph()
    # Find all connected components
    components = list(nx.connected_components(G))

    # Find the largest component by size
    giant_component = max(components, key=len)

    # Subgraph for the giant component
    giant_subgraph = G.subgraph(giant_component)

    ## the printing is for adding manuallyy the data to presentation
    print("giant_component:Nodes =", giant_subgraph.number_of_nodes())
    print("giant_component:Edges =", giant_subgraph.number_of_edges())

    return giant_subgraph

def average_distance():
    """
    Compute the average distance (average shortest path length) of the graph.

    Parameters:
    - G: networkx graph
    """
    G = build_graph()
    # Check if the graph is connected
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    else:
        return float('inf')  # Return infinity if the graph is disconnected

def calculate_edge_probabilities(G):
    # מציאת קהילות
    communities = list(greedy_modularity_communities(G))

    internal_probs = {}
    total_inter_edges = 0
    total_inter_possible = 0

    # עבור כל קהילה, חשב הסתברות פנימית
    for i, community in enumerate(communities):
        nodes = list(community)
        internal_edges = 0
        total_possible = 0

        for u, v in combinations(nodes, 2):
            total_possible += 1
            if G.has_edge(u, v):
                internal_edges += 1

        prob = internal_edges / total_possible if total_possible > 0 else 0
        internal_probs[f"Community_{i}"] = prob

    # חשב הסתברות בין קהילות
    for i in range(len(communities)):
        for j in range(i + 1, len(communities)):
            c1 = list(communities[i])
            c2 = list(communities[j])

            for u in c1:
                for v in c2:
                    total_inter_possible += 1
                    if G.has_edge(u, v):
                        total_inter_edges += 1

    external_prob = total_inter_edges / total_inter_possible if total_inter_possible > 0 else 0

    return internal_probs, external_prob


## up to non drafts ## TODO







# ----- Preferential Attachment -----

def plot_degree_distribution_log_binning(G, bin_count=20):
    degree_sequence = np.array([d for n, d in G.degree()])
    degree_count = Counter(degree_sequence)

    degrees = np.array(list(degree_count.keys()))
    counts = np.array(list(degree_count.values()))
    total = counts.sum()
    probabilities = counts / total

    # Logarithmic binning
    min_deg = degrees.min()
    max_deg = degrees.max()
    bins = np.logspace(np.log10(min_deg), np.log10(max_deg), bin_count)

    digitized = np.digitize(degrees, bins)
    binned_degrees = []
    binned_probs = []

    for i in range(1, len(bins)):
        bin_members = probabilities[digitized == i]
        bin_degrees = degrees[digitized == i]
        if len(bin_members) > 0:
            avg_prob = bin_members.mean()
            avg_degree = bin_degrees.mean()
            binned_degrees.append(avg_degree)
            binned_probs.append(avg_prob)

    # 3. Logarithmic binned plot
    plt.figure(figsize=(8, 5))
    plt.bar(binned_degrees, binned_probs, width=np.diff(bins).min(), align='center', color='orange')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Vertex Degree")
    plt.ylabel("Probability")
    plt.title("Logarithmic Binning (Reduced Noise)")
    plt.grid(True, which="both", ls="--")
    plt.show()

def build_pa_graph(original_graph):
    n = original_graph.number_of_nodes()
    m = int(original_graph.number_of_edges() / n)
    m = max(m, 1)  # המודל דורש m ≥ 1
    return nx.barabasi_albert_graph(n=n, m=m, seed=42)

def build_probabilistic_preferential_attachment_graph(initial_nodes=3, seed=None):
    """
    בונה גרף שבו כל צומת חדש מנסה להתחבר לכל הקיימים, על פי הסתברות פרופורציונלית לדרגה.

    :param n: מספר הקודקודים הסופי בגרף
    :param initial_nodes: מספר קודקודים להתחלה
    :param seed: לצורך שיחזור
    :return: גרף NetworkX
    """
    if seed is not None:
        random.seed(seed)

    G = build_graph()
    n = original.number_of_nodes()

    G.add_nodes_from(range(initial_nodes))
    # קישור ראשוני בין כל ההתחלתיים
    for i in range(initial_nodes):
        for j in range(i + 1, initial_nodes):
            G.add_edge(i, j)

    for new_node in range(initial_nodes, n):
        G.add_node(new_node)
        degrees = dict(G.degree())
        total_degree = sum(degrees.values())

        for existing_node in G.nodes():
            if existing_node == new_node:
                continue

            if total_degree == 0:
                if random.random() < 1 / (len(G.nodes()) - 1):
                    G.add_edge(new_node, existing_node)
            else:
                prob = degrees[existing_node] / total_degree
                if random.random() < prob:
                    G.add_edge(new_node, existing_node)

    return G


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
# generate_and_compare_PA_graph_with_hist()







# def (x):
#     '''
#     f(5 7)
#    x;
#
#     '''
# ----- התפלגות דרגות -----
# def plot_degree_distribution(G_real, G_pa):
#     def plot_hist(G, title):
#         degrees = [d for n, d in G.degree()]
#         plt.hist(degrees, bins=20, color='orange', edgecolor='black')
#         plt.xlabel("Degree")
#         plt.ylabel("Count")
#         plt.title(title)
#         plt.xscale("log")
#         plt.yscale("log")
#         plt.grid(True)
#
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plot_hist(G_real, "Original Graph (log-log)")
#     plt.subplot(1, 2, 2)
#     plot_hist(G_pa, "Preferential Attachment (log-log)")
#     plt.tight_layout()
#     plt.show()

# ----- התפלגות מצטברת ולינאריזציה לחוק חזקה -----
def loglog_fit_and_plot(G, label):
    degrees = [d for n, d in G.degree()]
    degree_count = Counter(degrees)
    deg, cnt = zip(*sorted(degree_count.items()))

    # התפלגות מצטברת
    total = sum(cnt)
    cum_prob = np.cumsum(cnt[::-1])[::-1] / total

    # log-log plot
    log_deg = np.log10(deg)
    log_cum_prob = np.log10(cum_prob)

    plt.scatter(log_deg, log_cum_prob, label=label, alpha=0.7)

    # התאמה לינארית
    slope, intercept, *_ = linregress(log_deg, log_cum_prob)
    fit_line = slope * np.array(log_deg) + intercept
    plt.plot(log_deg, fit_line, linestyle='--', label=f"{label} fit (α = {-slope:.2f})")

    return slope  # מחזיר את הערך של α



# --- e. Preferential Attachment model ---
def build_preferential_attachment_model():
    original = build_graph()
    n = original.number_of_nodes()
    m_total = original.number_of_edges()

    # לקבוע m כך שכל צומת חדש מחובר בערך למספר ממוצע של קשתות
    m = max(1, int(m_total / n))  # m חייב להיות לפחות 1 לפי דרישות הספרייה

    G = nx.barabasi_albert_graph(n, m, seed=42)
    return G


# Gpt distribution
def plot_degree_distribution_basic(G):
    degree_sequence = [d for n, d in G.degree()]
    degree_count = Counter(degree_sequence)

    degrees = sorted(degree_count.keys())
    counts = [degree_count[d] for d in degrees]

    # 1. Normal plot
    plt.figure(figsize=(8, 5))
    plt.bar(degrees, counts, color='orange')
    plt.xlabel("Vertex Degree")
    plt.ylabel("Count")
    plt.title("Degree Distribution")
    plt.grid(True)
    plt.show()

    # 2. Log-Log plot
    plt.figure(figsize=(8, 5))
    plt.bar(degrees, counts, color='orange')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Vertex Degree")
    plt.ylabel("Count")
    plt.title("Degree Distribution (Log-Log)")
    plt.grid(True, which="both", ls="--")
    plt.show()



# G = build_graph()
# plot_degree_distribution_basic(G)
# plot_degree_distribution_log_binning(G)


# hodaya
# def plot_distribution(degrees, title, filename, color, max_degree=None):
#     # Count degree frequencies
#     count = Counter(degrees)
#     count = {k: v for k, v in count.items() if k > 0}  # remove zero degrees (log undefined)
#     if max_degree:
#         count = {k: v for k, v in count.items() if k <= max_degree}
#
#     total = sum(count.values())
#     degs = np.array(sorted(count.keys()))
#     freqs = np.array([count[d] / total for d in degs])
#
#     # Take log of degrees
#     log_degs = np.log(degs)
#
#     # Fit a linear regression on log(degree) vs frequency
#     coeffs = np.polyfit(log_degs, freqs, 1)  # linear fit: y = a*log(x) + b
#     a, b = coeffs
#
#     # Generate smooth curve for fitted log function
#     x_smooth = np.linspace(min(degs), max(degs), 500)
#     y_smooth = a * np.log(x_smooth) + b
#
#     # Plotting
#     plt.figure(figsize=(10, 6))
#     plt.style.use('dark_background')
#     plt.scatter(degs, freqs, color=color, edgecolors='white', label='Data')
#     plt.plot(x_smooth, y_smooth, color='gold', linewidth=2, label='Log fit')
#
#     plt.title(title)
#     plt.xlabel("Degree")
#     plt.ylabel("Relative Frequency")
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(filename)
#     plt.close()
#     print(f"Saved: {filename}")
#
# def plot_normalized_degree_distributions_fixed(G):
#     # Collect the in-degree and out-degree values
#     in_degrees = [deg for _, deg in G.in_degree()]
#     out_degrees = [deg for _, deg in G.out_degree()]
#
#     # Plot the normalized degree distributions for both in-degree and out-degree
#     plot_distribution(in_degrees, "Normalized In-Degree Distribution", "normalized_in_degree_distribution.png",
#                       'royalblue') # , max_degree=50
#     plot_distribution(out_degrees, "Normalized Out-Degree Distribution", "normalized_out_degree_distribution.png",
#                       'tomato') # , max_degree=30

# G1 = build_graph()
# plot_degree_distribution(G1, normalized=False, log=False, title="Original Graph - Degree Distribution")
# plot_degree_distribution(G1, normalized=True, log=False, title="Original Graph - Normalized Degree Distribution")
# plot_degree_distribution(G1, normalized=True, log=True, title="Original Graph - Log-Log Degree Distribution")





# ----- הפעלה -----
if __name__ == "__main__":
    # draw_all_graphes()
    # plot_all_degree_distribution()
    #
    # draw_Graph(giant_component(),"Giant component")
    #
    # print("average_distance: ",average_distance())

    G = build_graph()
    check_power_law(G)

    check_powerlaw_builtin(G)





    # G_pa = build_pa_graph(G_original)
    #
    # # ציור התפלגות דרגות
    # plot_degree_distribution(G_original, G_pa)
    #
    # # ציור log-log של התפלגות מצטברת
    # plt.figure(figsize=(8, 6))
    # alpha_original = loglog_fit_and_plot(G_original, "Original Graph")
    # alpha_pa = loglog_fit_and_plot(G_pa, "Preferential Attachment")
    # plt.xlabel("log(Degree)")
    # plt.ylabel("log(Cumulative Probability)")
    # plt.title("Cumulative Degree Distribution (log-log)")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    #
    # print(f"Alpha (original graph): {-alpha_original:.4f}")
    # print(f"Alpha (PA graph):       {-alpha_pa:.4f}")



# draw_Graph(build_configuration_model(), "Configuration Model")
# G_pa = build_preferential_attachment_model()
# draw_Graph(G_pa, title="Preferential Attachment Model")



