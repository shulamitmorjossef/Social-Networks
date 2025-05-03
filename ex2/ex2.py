#
# """
# data = [
#     (1, 1, 3, 4, 5),
#     (2, 2, 4, 5, 6),
#     (3, 3, 5, 6, 7),
#     (4, 4, 6, 7, 8),
#     (5, 5, 7, 8, 9),
#     (6, 6, 8, 9, 10),
#     (7, 7, 9, 10, 11),
#     (8, 8, 10, 11, 12),
#     (9, 9, 11, 12, 1),
#     (10, 10, 12, 1, 2),
#     (11, 11, 1, 2, 3),
#     (12, 12, 2, 3, 4)
# ]
# """
#
#
import networkx as nx
import matplotlib.pyplot as plt

# הנתונים בפורמט רשימה של קשתות (ID, EdgeOne, EdgeTwo, EdgeThree, EdgeFour)
# data = [
#     (1, 2, 2, 2, 2),
#     (2, 2, 2, 2, 2),
#     (3, 2, 2, 2, 2),
#     (4, 4, 1, 2, 3),
# ]
data = [
    (1, 2, 2, 2, 2),
    (2, 2, 2, 2, 2),
    (3, 2, 2, 2, 2),
    (4, 4, 6, 7, 8),
    (5, 5, 7, 8, 9),
    (6, 6, 8, 9, 10),
    (7, 7, 9, 10, 11),
    (8, 8, 10, 11, 12),
    (9, 9, 11, 12, 1),
    (10, 10, 12, 1, 2),
    (11, 11, 1, 2, 3),
    (12, 12, 2, 3, 4)
]
# יצירת הגרף המכוון
G = nx.DiGraph()

# הוספת הקשתות לגרף
for row in data:
    node = row[0]
    for edge in row[1:]:
        G.add_edge(node, edge)

# חישוב ה-Pagerank
pagerank = nx.pagerank(G, alpha=0.85)

# הדפסת ה-Pagerank לכל קודקוד
maxx = 0
n = 0
print("PageRank values for each node:")
for node, score in pagerank.items():
    # print(f"Node {node}: {score}")
    if score > maxx:
        maxx = score
        n = node

print(f"Max PageRank value: {maxx}, node: {n}")

# print("\nDegree of each node:")
# for node in G.nodes:
#     in_deg = G.in_degree(node)
#     out_deg = G.out_degree(node)
#     print(f"Node {node}: In-degree = {in_deg}, Out-degree = {out_deg}")
#
#
# # ציור הגרף
# # plt.figure(figsize=(8, 6))
# # nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=12)
# # plt.show()
#
#
#
# # 0.6442331149735718:
# # data = [
# #     (1, 2, 2, 2, 2),
# #     (2, 2, 2, 2, 2),
# #     (3, 2, 2, 2, 2),
# #     (4, 4, 6, 7, 8),
# #     (5, 5, 7, 8, 9),
# #     (6, 6, 8, 9, 10),
# #     (7, 7, 9, 10, 11),
# #     (8, 8, 10, 11, 12),
# #     (9, 9, 11, 12, 1),
# #     (10, 10, 12, 1, 2),
# #     (11, 11, 1, 2, 3),
# #     (12, 12, 2, 3, 4)
# # ]
#
#
# import itertools
# import networkx as nx
#
#
# def compute_pagerank(data):
#     G = nx.DiGraph()
#     for row in data:
#         node = row[0]
#         for edge in row[1:]:
#             G.add_edge(node, edge)
#     pagerank = nx.pagerank(G, alpha=0.85)
#     return pagerank
#
#
# def generate_combinations(data):
#     nodes = [row[0] for row in data]  # רשימת הקודקודים האפשריים
#     combinations = list(itertools.combinations(nodes, 3))  # כל צירוף של 3 קודקודים
#     results = []
#
#     for target in nodes:
#         for comb in combinations:
#             modified_data = [list(row) for row in data]
#             for row in modified_data:
#                 if row[0] in comb:
#                     row[1:] = [target] * 4  # כל השלושה יצביעו על אותו קודקוד
#
#             pagerank = compute_pagerank(modified_data)
#             max_node = max(pagerank, key=pagerank.get)
#             max_value = pagerank[max_node]
#
#             results.append((comb, target, max_node, max_value))
#
#     return sorted(results, key=lambda x: -x[3])  # מיון לפי הדירוג הגבוה ביותר
#
#
# # הנתונים המקוריים
# data = [
#     (1, 1, 3, 4, 5),
#     (2, 2, 4, 5, 6),
#     (3, 3, 5, 6, 7),
#     (4, 4, 6, 7, 8),
#     (5, 5, 7, 8, 9),
#     (6, 6, 8, 9, 10),
#     (7, 7, 9, 10, 11),
#     (8, 8, 10, 11, 12),
#     (9, 9, 11, 12, 1),
#     (10, 10, 12, 1, 2),
#     (11, 11, 1, 2, 3),
#     (12, 12, 2, 3, 4)
# ]
#
# best_configurations = generate_combinations(data)
# print("Top configurations:")
# for comb, target, max_node, max_value in best_configurations[:5]:
#     print(f"Nodes {comb} all pointing to {target} -> Max PageRank at {max_node} with {max_value}")
