from collections import Counter
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ----- קריאה לקבצים -----
file = pd.read_csv("stormofswords.csv").iloc[1:]
tribes = pd.read_csv("tribes.csv")

nodes = tribes.iloc[:, 0].tolist()
edges = list(zip(file["Source"], file["Target"], file["Weight"]))

# ----- בניית הגרף המקורי -----
def build_graph():
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for src, tgt, weight in edges:
        G.add_edge(src, tgt, weight=weight)
    return G

def draw_Graph(G, title="Graph"):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=False, node_color='lightblue', edge_color='gray', node_size=500)
    plt.title(title)
    plt.show()

# ----- גרף Preferential Attachment -----
def build_pa_graph(original_graph):
    n = original_graph.number_of_nodes()
    m = int(original_graph.number_of_edges() / n)  # ממוצע חיבורים פר צומת
    m = max(m, 1)  # המודל דורש m ≥ 1
    return nx.barabasi_albert_graph(n=n, m=m, seed=42)

# ----- התפלגות דרגות -----
def plot_degree_distribution(G_real, G_pa):
    def plot_hist(G, title):
        degrees = [d for n, d in G.degree()]
        plt.hist(degrees, bins=20, color='orange', edgecolor='black')
        plt.xlabel("Degree")
        plt.ylabel("Count")
        plt.title(title)
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_hist(G_real, "Original Graph (log-log)")
    plt.subplot(1, 2, 2)
    plot_hist(G_pa, "Preferential Attachment (log-log)")
    plt.tight_layout()
    plt.show()

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
    G_original = build_graph()
    G_pa = build_pa_graph(G_original)

    # ציור התפלגות דרגות
    plot_degree_distribution(G_original, G_pa)

    # ציור log-log של התפלגות מצטברת
    plt.figure(figsize=(8, 6))
    alpha_original = loglog_fit_and_plot(G_original, "Original Graph")
    alpha_pa = loglog_fit_and_plot(G_pa, "Preferential Attachment")
    plt.xlabel("log(Degree)")
    plt.ylabel("log(Cumulative Probability)")
    plt.title("Cumulative Degree Distribution (log-log)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Alpha (original graph): {-alpha_original:.4f}")
    print(f"Alpha (PA graph):       {-alpha_pa:.4f}")

#
# draw_Graph(build_graph())
# draw_Graph(build_gnm_graph(), "G(n, m) Graph")
# draw_Graph(build_configuration_model(), "Configuration Model")
# G_pa = build_preferential_attachment_model()
# draw_Graph(G_pa, title="Preferential Attachment Model")
