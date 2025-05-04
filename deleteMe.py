# ------------------------------------------------
# ייבוא ספריות (מודולים) שדרושות לקוד
# ------------------------------------------------
import networkx as nx               # גרפים ורשתות
import pandas as pd                # קריאת קבצי CSV ועיבוד טבלאות
import numpy as np                 # חישובים מתמטיים
import matplotlib.pyplot as plt    # גרפים וציורים
import powerlaw                   # בדיקת התפלגות חוק חזקה

# ------------------------------------------------
# 1. טעינת רשת Game of Thrones
# ------------------------------------------------
# קובץ CSV שבו מופיעים חיבורים בין דמויות (Source-Target)
df = pd.read_csv("stormofswords.csv")
# בניית גרף מהטבלה - כל שורה היא קשת בין שני קודקודים
G_original = nx.from_pandas_edgelist(df, 'Source', 'Target')

# שמירת מספר הקודקודים והקשתות ברשת
n = G_original.number_of_nodes()
m = G_original.number_of_edges()
print(f"Original network: {n} nodes, {m} edges")

# ------------------------------------------------
# 2. יצירת גרפים אקראיים לפי מודלים שונים
# ------------------------------------------------

# a. מודל G(n,p) של ארדוש-רני - כל זוג קודקודים מחובר בהסתברות p
p = 0.01
G_np = nx.erdos_renyi_graph(n, p)

# b. מודל G(n,m) - גרף עם מספר קודקודים n ומספר קשתות m קבוע
G_nm = nx.gnm_random_graph(n, m/5)  # משתמשים רק בחמישית מהקשתות של המקורי

# c. מודל תצורת דרגות (Configuration Model)
# שמירה על דרגות הקודקודים מהגרף המקורי ויצירת גרף חדש עם אותן דרגות
degree_sequence = [d for _, d in G_original.degree()]
G_config = nx.configuration_model(degree_sequence)
G_config = nx.Graph(G_config)  # הסרת קשתות כפולות
G_config.remove_edges_from(nx.selfloop_edges(G_config))  # הסרת לולאות עצמיות

# ציור גרפי להשוואה בין הגרף המקורי לבין מודל התצורה
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
nx.draw(G_original, with_labels=True, node_color='lightblue')
plt.title("Original Graph")

plt.subplot(1, 2, 2)
nx.draw(G_config, with_labels=True, node_color='lightgreen')
plt.title("Configuration Model Graph")
plt.tight_layout()
plt.show()

# d. מודל בלוקים (קהילות) פשוט
# מחלקים את הרשת לשני בלוקים בגודל שונה עם הסתברויות פנימיות וחיצוניות
sizes = [n // 2, n - n // 2]  # גודל כל בלוק
p_in = 0.1     # הסתברות לחיבור בתוך בלוק
p_out = 0.02   # הסתברות לחיבור בין בלוקים
probs = [[p_in, p_out], [p_out, p_in]]  # מטריצת הסתברויות
G_block = nx.stochastic_block_model(sizes, probs)

# יצירת גרף בלוקים לפי חלוקה לשבטים מתוך קובץ tribes.csv
df = pd.read_csv("tribes.csv")  # הקובץ כולל עמודות: node_id, tribe
tribes = df.groupby('tribe')['node_id'].apply(list).to_dict()  # קיבוץ לפי שבט

# גודל כל שבט
sizes = [len(tribes[t]) for t in tribes]

# מטריצת הסתברויות: הסתברות גבוהה בתוך שבט (0.8), נמוכה בין שבטים (0.05)
num_tribes = len(sizes)
probs = [[0.8 if i == j else 0.05 for j in range(num_tribes)] for i in range(num_tribes)]

# יצירת הגרף לפי שבטים
G = nx.stochastic_block_model(sizes, probs)

# צביעת הקודקודים לפי השבט שהם שייכים אליו
community_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
node_colors = []
for block_idx, size in enumerate(sizes):
    node_colors.extend([community_colors[block_idx % len(community_colors)]] * size)

# ציור הגרף לפי צבעים של שבטים
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='lightgreen')
plt.title("Graph by Tribes")
plt.show()

# e. מודל העדפה מצטברת (Barabasi-Albert)
# כל קודקוד חדש מתחבר ל-m קודקודים קיימים עם הסתברות פרופורציונלית לדרגה
m_ba = max(1, int(np.mean(degree_sequence) // 2))
G_ba = nx.barabasi_albert_graph(n, m_ba)

# ------------------------------------------------
# 3. ציור גרפי של כל הרשתות שנוצרו
# ------------------------------------------------
all_graphs = {
    "Original Network": G_original,
    "G(n,p) Random Graph": G_np,
    "G(n,m) Random Graph": G_nm,
    "Configuration Model": G_config,
    "Block Model": G_block,
    "Preferential Attachment (BA)": G_ba
}

# ציור לכל גרף בעזרת פריסת spring layout
for name, G in all_graphs.items():
    plt.figure(figsize=(8,8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(G, pos, node_size=40, with_labels=False, edge_color="gray")
    plt.title(f"Network Visualization: {name}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ------------------------------------------------
# 4. ציור התפלגות הדרגות של כל רשת
# ------------------------------------------------
for name, G in all_graphs.items():
    plt.figure(figsize=(7,5))
    degrees = [d for _, d in G.degree()]
    plt.hist(degrees, bins=range(1, max(degrees)+2), alpha=0.7, color="royalblue", edgecolor="black")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title(f"Degree Distribution: {name}")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------
# 5. השוואת הרשתות לפי רכיב קשיר ומרחק ממוצע
# ------------------------------------------------

# פונקציה למציאת גודל הרכיב הקשיר הגדול ביותר
def giant_component_size(G):
    return len(max(nx.connected_components(G), key=len))

# פונקציה לחישוב מרחק ממוצע - רק על הרכיב הקשיר הגדול אם צריך
def average_distance(G):
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        return nx.average_shortest_path_length(G.subgraph(largest_cc))

# פונקציה למציאת דרגה ממוצעת
def average_degree(G):
    return np.mean([d for _, d in G.degree()])

# הדפסת טבלה עם מידע על כל גרף
print(f"{'Network':<30} {'Giant Component':<18} {'Avg Distance':<18} {'Avg Degree':<12}")
for name, G in all_graphs.items():
    gc = giant_component_size(G)
    ad = average_distance(G)
    avg_deg = average_degree(G)
    print(f"{name:<30} {gc:<18} {ad:<18.2f} {avg_deg:<12.2f}")

# ------------------------------------------------
# 6. בדיקת חוק חזקה (Power Law) על הרשת המקורית
# ------------------------------------------------
# ניתוח התפלגות הדרגות בעזרת הספרייה powerlaw
G_my = G_original
degrees = np.array([d for _, d in G_my.degree()])

# התאמה לחוק חזקה - ניתוח רק לדרגות > 0
fit = powerlaw.Fit(degrees[degrees > 0])

# ציור גרף: התפלגות הדרגות לעומת התאמה לחוק חזקה
plt.figure()
fit.plot_pdf(label='Empirical')
fit.power_law.plot_pdf(color='r', linestyle='--', label='Power law fit')
plt.xlabel("Degree")
plt.ylabel("P(k)")
plt.legend()
plt.title("Power-law Degree Distribution (Your Network)")
plt.show()

# הדפסת האקספוננט של חוק החזקה
print(f"Estimated power-law exponent (beta): {fit.power_law.alpha:.2f}")

# ------------------------------------------------
# 7. השוואה בין הרשת המקורית לרשת שנוצרה לפי מודל BA
# ------------------------------------------------
G_model = G_ba

# השוואת התפלגות הדרגות
plt.figure(figsize=(8,5))
degrees_my = [d for _, d in G_my.degree()]
degrees_model = [d for _, d in G_model.degree()]
plt.hist(degrees_my, bins=range(1, max(degrees_my)+2), alpha=0.5, label="OTC", density=True)
plt.hist(degrees_model, bins=range(1, max(degrees_model)+2), alpha=0.5, label="BA Model", density=True)
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.legend()
plt.title("Degree Distribution: OTC vs. BA Model")
plt.show()

# השוואת מדדים בין הרשתות
print(f"OTC - Giant Component: {giant_component_size(G_my)}")
print(f"OTC - Average Distance: {average_distance(G_my):.2f}")
print(f"BA Model - Giant Component: {giant_component_size(G_model)}")
print(f"BA Model - Average Distance: {average_distance(G_model):.2f}")
