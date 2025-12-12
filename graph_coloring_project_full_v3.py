import random
import time
from collections import defaultdict
import csv
import os
import math

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec


# =========================================================
# Graph Coloring Project - Full Integrated Code (v5+gap table)
# =========================================================
#   - Graph structure & generators
#   - Greedy, Welsh-Powell, DSatur, Exact Backtracking
#   - Stress-test graph where Greedy is clearly worse
#   - Experiment runner (CSV results)
#   - Per-instance bar plots (colors & runtime, log-scale)
#   - Colored graph visualizations (DSatur vs Exact)
#   - Stress comparison figure (Greedy vs DSatur)
#   - Chromatic gap summary plot
#   - Colors vs n summary plot
#   - Time vs n summary plot
#   - NEW: Chromatic gap summary table + combined table+chart figure
# =========================================================


# -----------------------------
# Global plotting style
# -----------------------------

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.figsize": (6.5, 4.5),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "axes.facecolor": "#f7f7f9",
        "axes.edgecolor": "#444444",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "legend.frameon": False,
    }
)

# Two main accent colors for bar charts (no yellow)
BAR_COLOR_COLORS = "#4C72B0"  # deeper blue
BAR_COLOR_TIMES = "#55A868"   # deeper green

# Consistent colors for algorithms in summary plots
ALG_COLOR_MAP = {
    "Greedy (input order)": "#4C72B0",  # blue
    "Welsh-Powell": "#55A868",          # green
    "DSatur": "#C44E52",                # red
}


# -----------------------------
# Graph utilities
# -----------------------------


class Graph:
    """Simple undirected graph using adjacency sets."""

    def __init__(self):
        self.adj = defaultdict(set)

    def add_edge(self, u, v):
        if u == v:
            return
        self.adj[u].add(v)
        self.adj[v].add(u)

    def add_vertex(self, v):
        _ = self.adj[v]

    def vertices(self):
        return list(self.adj.keys())

    def neighbors(self, v):
        return self.adj[v]

    def degree(self, v):
        return len(self.adj[v])

    def __len__(self):
        return len(self.adj)


# -----------------------------
# Graph generators
# -----------------------------


def complete_graph(n):
    """Complete graph K_n."""
    g = Graph()
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(i, j)
    return g


def bipartite_complete(n1, n2):
    """Complete bipartite graph K_{n1, n2}."""
    g = Graph()
    for i in range(n1):
        for j in range(n2):
            g.add_edge(("A", i), ("B", j))
    return g


def erdos_renyi_graph(n, p, seed=None):
    """Erdos-Renyi random graph G(n, p)."""
    if seed is not None:
        random.seed(seed)
    g = Graph()
    for i in range(n):
        g.add_vertex(i)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                g.add_edge(i, j)
    return g


# -----------------------------
# Coloring helpers
# -----------------------------


def is_valid_coloring(graph, coloring):
    for v in graph.vertices():
        for u in graph.neighbors(v):
            if coloring.get(v) == coloring.get(u):
                return False
    return True


def color_count(coloring):
    return len(set(coloring.values())) if coloring else 0


# -----------------------------
# Coloring algorithms
# -----------------------------


def greedy_coloring(graph, order=None):
    """Simple greedy coloring given a vertex order.

    This is the 'Baseline Greedy (input order)' algorithm.
    """
    if order is None:
        order = graph.vertices()
    coloring = {}
    for v in order:
        used = {coloring[u] for u in graph.neighbors(v) if u in coloring}
        color = 1
        while color in used:
            color += 1
        coloring[v] = color
    return coloring


def welsh_powell_coloring(graph):
    """Welsh-Powell heuristic: sort vertices by degree descending, then greedy."""
    order = sorted(graph.vertices(), key=lambda v: graph.degree(v), reverse=True)
    return greedy_coloring(graph, order)


def dsatur_coloring(graph):
    """DSatur heuristic coloring."""
    vertices = graph.vertices()
    n = len(vertices)
    if n == 0:
        return {}

    degrees = {v: graph.degree(v) for v in vertices}
    saturation = {v: 0 for v in vertices}
    coloring = {}

    # Start with a highest-degree vertex
    current = max(vertices, key=lambda v: degrees[v])
    coloring[current] = 1

    while len(coloring) < n:
        # Update saturation for neighbors of the last colored vertex
        for u in graph.neighbors(current):
            if u not in coloring:
                neighbor_colors = {
                    coloring[v] for v in graph.neighbors(u) if v in coloring
                }
                saturation[u] = len(neighbor_colors)

        # Select next vertex: highest saturation, break ties by degree
        uncolored = [v for v in vertices if v not in coloring]
        current = max(uncolored, key=lambda v: (saturation[v], degrees[v]))

        # Assign the smallest available color
        neighbor_colors = {coloring[v] for v in graph.neighbors(current) if v in coloring}
        color = 1
        while color in neighbor_colors:
            color += 1
        coloring[current] = color

    return coloring


def exact_backtracking_coloring(graph, max_time=5.0):
    """Exact coloring using backtracking + simple branch and bound.

    Only suitable for small graphs (e.g., n <= 30).
    Returns (coloring_dict, number_of_colors, elapsed_time, timeout_flag).
    """
    vertices = sorted(graph.vertices(), key=lambda v: graph.degree(v), reverse=True)
    n = len(vertices)
    if n == 0:
        return {}, 0, 0.0, False

    # Use DSatur as an initial upper bound (good pruning)
    start = time.perf_counter()
    ds_col = dsatur_coloring(graph)
    best_coloring = dict(ds_col)
    best_k = color_count(ds_col)

    timeout = False

    def backtrack(idx, coloring, used_colors):
        nonlocal best_coloring, best_k, timeout
        # Check timeout
        if time.perf_counter() - start > max_time:
            timeout = True
            return

        if idx == n:
            k = len(used_colors)
            if k < best_k:
                best_k = k
                best_coloring = dict(coloring)
            return

        # Branch & bound: if we already use >= best_k colors, no need to go deeper
        if len(used_colors) >= best_k:
            return

        v = vertices[idx]

        # Try existing colors first
        for c in range(1, best_k + 1):
            if c in used_colors:
                if all(coloring.get(u) != c for u in graph.neighbors(v) if u in coloring):
                    coloring[v] = c
                    backtrack(idx + 1, coloring, used_colors)
                    del coloring[v]
                    if timeout:
                        return

        # Try a new color if it could give a better solution
        new_color = len(used_colors) + 1
        if new_color < best_k:
            if all(coloring.get(u) != new_color for u in graph.neighbors(v) if u in coloring):
                coloring[v] = new_color
                backtrack(idx + 1, coloring, used_colors | {new_color})
                del coloring[v]

    backtrack(0, {}, set(best_coloring.values()))
    elapsed = time.perf_counter() - start
    return best_coloring, best_k, elapsed, timeout


# -----------------------------
# Experiment helpers
# -----------------------------


def build_instances():
    """Create a set of benchmark graphs for experiments."""
    instances = []

    # Small random graphs (exact feasible)
    instances.append(
        {
            "name": "small_er_n20_p02",
            "graph": erdos_renyi_graph(20, 0.2, seed=1),
            "category": "small_random",
            "optimal_known": False,
        }
    )
    instances.append(
        {
            "name": "small_er_n25_p03",
            "graph": erdos_renyi_graph(25, 0.3, seed=2),
            "category": "small_random",
            "optimal_known": False,
        }
    )

    # Medium random graphs (exact may be infeasible)
    instances.append(
        {
            "name": "medium_er_n80_p02",
            "graph": erdos_renyi_graph(80, 0.2, seed=3),
            "category": "medium_sparse",
            "optimal_known": False,
        }
    )
    instances.append(
        {
            "name": "medium_er_n80_p05",
            "graph": erdos_renyi_graph(80, 0.5, seed=4),
            "category": "medium_dense",
            "optimal_known": False,
        }
    )

    # Special structure: bipartite and complete
    instances.append(
        {
            "name": "bipartite_20_20",
            "graph": bipartite_complete(20, 20),
            "category": "bipartite",
            "optimal_known": True,
            "optimal_chi": 2,
        }
    )
    instances.append(
        {
            "name": "complete_n20",
            "graph": complete_graph(20),
            "category": "complete",
            "optimal_known": True,
            "optimal_chi": 20,
        }
    )

    # Stress-test instance:
    instances.append(
        {
            "name": "stress_greedy_vs_dsatur",
            "graph": erdos_renyi_graph(15, 0.5, seed=844),
            "category": "stress",
            "optimal_known": False,
        }
    )

    return instances


def time_algorithm(alg, graph, repeat=1):
    start = time.perf_counter()
    coloring = None
    for _ in range(repeat):
        coloring = alg(graph)
    elapsed = (time.perf_counter() - start) / repeat
    return coloring, elapsed


def run_experiments(output_csv="results_graph_coloring.csv"):
    instances = build_instances()

    # Human-readable algorithm names for CSV & plots
    algorithms = {
        "Greedy (input order)": greedy_coloring,
        "Welsh-Powell": welsh_powell_coloring,
        "DSatur": dsatur_coloring,
    }

    results = []

    fieldnames = [
        "instance",
        "category",
        "n_vertices",
        "algorithm",
        "colors",
        "time_sec",
        "optimal_known",
        "optimal_chi",
        "error",
        "valid_coloring",
        "exact_colors",
        "exact_time_sec",
        "exact_timeout",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for inst in instances:
            name = inst["name"]
            g = inst["graph"]
            category = inst["category"]
            n = len(g)
            optimal_known = inst.get("optimal_known", False)
            optimal_chi = inst.get("optimal_chi", None)

            print(f"\n=== Instance {name} (|V|={n}, category={category}) ===")

            # Exact algorithm only for small graphs (n <= 30)
            exact_colors = None
            exact_time = None
            exact_timeout = False
            if n <= 30:
                print("Running exact backtracking (small graph)...")
                exact_coloring, exact_colors, exact_time, exact_timeout = (
                    exact_backtracking_coloring(g)
                )
                print(
                    f"  Exact: colors={exact_colors}, time={exact_time:.4f}s, "
                    f"timeout={exact_timeout}"
                )
            else:
                exact_coloring = None

            for alg_name, alg_fun in algorithms.items():
                print(f"Running {alg_name}...")
                coloring, t = time_algorithm(alg_fun, g)
                k = color_count(coloring)
                valid = is_valid_coloring(g, coloring)

                # Error relative to known optimum (if available from exact or theory)
                error = None
                opt_used = None
                if exact_colors is not None and not exact_timeout:
                    error = k - exact_colors
                    opt_used = exact_colors
                elif optimal_known and optimal_chi is not None:
                    error = k - optimal_chi
                    opt_used = optimal_chi

                row = {
                    "instance": name,
                    "category": category,
                    "n_vertices": n,
                    "algorithm": alg_name,
                    "colors": k,
                    "time_sec": t,
                    "optimal_known": bool(opt_used is not None),
                    "optimal_chi": opt_used,
                    "error": error,
                    "valid_coloring": valid,
                    "exact_colors": exact_colors,
                    "exact_time_sec": exact_time,
                    "exact_timeout": exact_timeout,
                }
                writer.writerow(row)
                results.append(row)

    print(f"\nResults saved to {output_csv}")
    return results


# -----------------------------
# Plotting helpers
# -----------------------------


def _annotate_horizontal_bars(ax, values, is_time=False):
    for bar, val in zip(ax.patches, values):
        if is_time:
            label = f"{val:.2e}"  # 科学计数法显示时间
        else:
            label = f"{val:.2f}" if isinstance(val, float) else str(val)

        ax.text(
            bar.get_width() * 1.01,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=10,
        )


def plot_for_instance(results, instance_name, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    subset = [r for r in results if r["instance"] == instance_name]
    if not subset:
        print(f"No results for instance {instance_name}")
        return

    algs = [r["algorithm"] for r in subset]
    colors_used = [r["colors"] for r in subset]
    times = [r["time_sec"] for r in subset]

    # Horizontal bar chart for colors
    fig, ax = plt.subplots()
    y_positions = range(len(algs))
    ax.barh(
        list(y_positions),
        colors_used,
        color=BAR_COLOR_COLORS,
        edgecolor="black",
        alpha=0.9,
    )
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(algs)
    ax.set_xlabel("Number of Colors")
    ax.set_title(f"Colors Used - {instance_name}")
    _annotate_horizontal_bars(ax, colors_used)
    fig.tight_layout()
    colors_path = os.path.join(out_dir, f"{instance_name}_colors.png")
    fig.savefig(colors_path, dpi=300)
    plt.show()
    plt.close(fig)

    # Horizontal bar chart for runtimes (log scale on x-axis)
    fig, ax = plt.subplots()
    times_log = [max(t, 1e-5) for t in times]  # avoid 0 on log scale
    ax.barh(
        list(y_positions),
        times_log,
        color=BAR_COLOR_TIMES,
        edgecolor="black",
        alpha=0.9,
    )
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(algs)
    ax.set_xlabel("Runtime (seconds, log scale)")
    ax.set_xscale("log")
    ax.set_title(f"Runtime Comparison - {instance_name}")
    _annotate_horizontal_bars(ax, times, is_time=True)
    fig.tight_layout()
    times_path = os.path.join(out_dir, f"{instance_name}_runtime.png")
    fig.savefig(times_path, dpi=300)
    plt.show()
    plt.close(fig)

    print(f"Saved plots for {instance_name} to {colors_path} and {times_path}")


def plot_all_instances(results, out_dir="plots"):
    instance_names = sorted(set(r["instance"] for r in results))
    for name in instance_names:
        plot_for_instance(results, name, out_dir=out_dir)


# -----------------------------
# Colored graph visualization
# -----------------------------


def _compute_circle_positions(vertices):
    n = len(vertices)
    positions = {}
    for i, v in enumerate(vertices):
        angle = 2 * math.pi * i / n
        x = math.cos(angle)
        y = math.sin(angle)
        positions[v] = (x, y)
    return positions


def _get_palette(palette, used_colors):
    if palette == "exact":
        base_colors = [
            "#1F77B4",
            "#2CA02C",
            "#D62728",
            "#9467BD",
            "#8C564B",
            "#E377C2",
            "#7F7F7F",
            "#17BECF",
        ]
    else:  # dsatur / default
        base_colors = [
            "#4C72B0",
            "#55A868",
            "#C44E52",
            "#8172B2",
            "#937860",
            "#8C8C8C",
        ]

    color_map = {}
    for i, c in enumerate(sorted(used_colors)):
        color_map[c] = base_colors[i % len(base_colors)]
    return color_map


def _draw_colored_graph_on_axes(ax, graph, coloring, palette="dsatur", show_labels=True):
    vertices = graph.vertices()
    if not vertices:
        return

    positions = _compute_circle_positions(vertices)
    used_colors = set(coloring.values())
    color_map = _get_palette(palette, used_colors)

    # Draw edges
    for v in vertices:
        x1, y1 = positions[v]
        for u in graph.neighbors(v):
            if str(u) > str(v):  # avoid drawing twice
                x2, y2 = positions[u]
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    linewidth=0.6,
                    color="#d0d0e0",
                    zorder=1,
                )

    # Draw vertices
    for v in vertices:
        x, y = positions[v]
        c = coloring.get(v, 0)
        ax.scatter(
            x,
            y,
            s=260,
            color=color_map.get(c, "#333333"),
            edgecolors="#222222",
            linewidths=0.8,
            zorder=2,
        )
        if show_labels:
            ax.text(
                x,
                y,
                str(v),
                fontsize=8,
                ha="center",
                va="center",
                color="white",
                zorder=3,
            )
    ax.set_axis_off()


def draw_colored_graph(graph, coloring, title, out_path, palette="dsatur"):
    """Standalone figure for coloring visualization."""
    fig, ax = plt.subplots(figsize=(5, 5))
    _draw_colored_graph_on_axes(ax, graph, coloring, palette=palette)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved colored graph to {out_path}")


def create_colored_graphs(instances, out_dir="colored_graphs"):
    os.makedirs(out_dir, exist_ok=True)
    for inst in instances:
        name = inst["name"]
        g = inst["graph"]
        n = len(g)

        # Only visualize relatively small graphs to keep the picture readable
        if n <= 25 or inst["category"] in ["bipartite", "complete", "stress"]:
            # 1) DSatur heuristic coloring
            col_ds = dsatur_coloring(g)
            k_ds = color_count(col_ds)
            title_ds = f"{name} (DSatur, {k_ds} colors)"
            out_ds = os.path.join(out_dir, f"{name}_dsatur_coloring.png")
            draw_colored_graph(g, col_ds, title_ds, out_ds, palette="dsatur")

            # 2) Exact minimum coloring (if available and not timed out)
            if n <= 30:
                exact_col, exact_k, exact_t, timeout = exact_backtracking_coloring(g)
                if exact_col is not None and not timeout:
                    title_ex = f"{name} (Exact minimum, {exact_k} colors)"
                    out_ex = os.path.join(out_dir, f"{name}_exact_coloring.png")
                    draw_colored_graph(
                        g, exact_col, title_ex, out_ex, palette="exact"
                    )


# -----------------------------
# Stress comparison figure
# -----------------------------


def plot_stress_comparison(instances, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    # Find the stress instance
    stress = None
    for inst in instances:
        if inst["name"] == "stress_greedy_vs_dsatur":
            stress = inst
            break
    if stress is None:
        print("No stress instance found.")
        return

    g = stress["graph"]

    col_greedy = greedy_coloring(g)
    col_dsatur = dsatur_coloring(g)
    k_g = color_count(col_greedy)
    k_d = color_count(col_dsatur)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    _draw_colored_graph_on_axes(
        axes[0], g, col_greedy, palette="dsatur", show_labels=True
    )
    axes[0].set_title(f"Greedy (input order)\n{k_g} colors")

    _draw_colored_graph_on_axes(
        axes[1], g, col_dsatur, palette="exact", show_labels=True
    )
    axes[1].set_title(f"DSatur\n{k_d} colors")

    fig.suptitle("Stress Instance: Greedy vs DSatur", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(out_dir, "stress_greedy_vs_dsatur_comparison.png")
    fig.savefig(out_path, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved stress comparison figure to {out_path}")


# -----------------------------
# Chromatic gap summary plot
# -----------------------------


def plot_chromatic_gap_summary(results, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)

    # We only consider rows where optimal_known is True and error is not None
    filtered = [r for r in results if r["optimal_known"] and r["error"] is not None]
    if not filtered:
        print("No instances with known optimal solution for gap plot.")
        return

    # Collect instances and algorithms
    instances = sorted(set(r["instance"] for r in filtered))
    algorithms = ["Greedy (input order)", "Welsh-Powell", "DSatur"]

    # Build matrix: rows = instances, cols = algorithms
    gap_matrix = [[0.0 for _ in algorithms] for _ in instances]

    for r in filtered:
        i_idx = instances.index(r["instance"])
        a_idx = algorithms.index(r["algorithm"])
        gap_matrix[i_idx][a_idx] = r["error"]

    # Plot grouped bar chart
    x = range(len(instances))
    bar_width = 0.22

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for j, alg in enumerate(algorithms):
        offsets = [xi + (j - 1) * bar_width for xi in x]
        gaps = [gap_matrix[i][j] for i in range(len(instances))]
        ax.bar(
            offsets,
            gaps,
            width=bar_width,
            label=alg,
            color=ALG_COLOR_MAP.get(alg, None),
            edgecolor="black",
            alpha=0.9,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(instances, rotation=20)
    ax.set_ylabel("Gap to Optimal (#colors)")
    ax.set_title("Chromatic Gap (Heuristic Colors - Optimal Colors)")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(out_dir, "summary_chromatic_gap.png")
    fig.savefig(out_path, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved chromatic gap summary to {out_path}")


# -----------------------------
# Chromatic gap table + combined figure
# -----------------------------


def plot_gap_table_and_chart(results, out_dir="plots"):
    """Create a gap summary table (CSV + markdown) and a combined table+chart figure."""
    os.makedirs(out_dir, exist_ok=True)

    filtered = [r for r in results if r["optimal_known"] and r["error"] is not None]
    if not filtered:
        print("No gap data available for table.")
        return

    # Build dict: instance -> {algorithm -> gap}
    algorithms = ["Greedy (input order)", "Welsh-Powell", "DSatur"]
    table_dict = {}

    for r in filtered:
        inst = r["instance"]
        alg = r["algorithm"]
        gap = r["error"]
        if inst not in table_dict:
            table_dict[inst] = {a: 0 for a in algorithms}
        table_dict[inst][alg] = gap

    # Build DataFrame with fixed column names for clarity
    instances_sorted = sorted(table_dict.keys())
    rows = []
    for inst in instances_sorted:
        gaps = table_dict[inst]
        rows.append(
            [
                gaps["Greedy (input order)"],
                gaps["Welsh-Powell"],
                gaps["DSatur"],
            ]
        )

    table_df = pd.DataFrame(
        rows,
        index=instances_sorted,
        columns=["Greedy", "Welsh-Powell", "DSatur"],
    ).astype(int)

    # Save table as CSV & Markdown
    table_df.to_csv(os.path.join(out_dir, "chromatic_gap_table.csv"))
    table_df.to_markdown(os.path.join(out_dir, "chromatic_gap_table.md"))
    print("\n=== Chromatic Gap Table ===")
    print(table_df)

    # --- Combined figure: Table (left) + Chart (right) ---
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, width_ratios=[1, 1.4])

    # Left: Table
    ax_table = fig.add_subplot(gs[0])
    ax_table.axis("off")
    table_display = ax_table.table(
        cellText=table_df.values,
        rowLabels=table_df.index,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    table_display.auto_set_font_size(False)
    table_display.set_fontsize(11)
    table_display.scale(1.1, 1.2)

    # Right: Grouped bar chart
    ax_chart = fig.add_subplot(gs[1])

    x = range(len(table_df.index))
    bar_width = 0.25

    alg_pairs = [
        ("Greedy", "Greedy (input order)"),
        ("Welsh-Powell", "Welsh-Powell"),
        ("DSatur", "DSatur"),
    ]

    for j, (col_name, label_name) in enumerate(alg_pairs):
        gaps = table_df[col_name].values
        offsets = [xi + (j - 1) * bar_width for xi in x]
        ax_chart.bar(
            offsets,
            gaps,
            width=bar_width,
            label=label_name,
            color=ALG_COLOR_MAP.get(label_name, "#777777"),
            edgecolor="black",
            alpha=0.9,
        )

    ax_chart.set_xticks(list(x))
    ax_chart.set_xticklabels(table_df.index, rotation=25, ha="right")
    ax_chart.set_ylabel("Gap to Optimal (#colors)")
    ax_chart.set_title("Chromatic Gap Bar Chart")
    ax_chart.axhline(0, color="#333333", linewidth=1)
    ax_chart.legend()

    fig.suptitle("Chromatic Gap Summary (Table + Chart)", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(out_dir, "gap_table_and_chart.png")
    fig.savefig(out_path, dpi=300)
    plt.show()
    plt.close(fig)

    print(f"Saved combined gap table + chart to {out_path}")


# -----------------------------
# Colors vs n & Time vs n summary plots
# -----------------------------


def plot_colors_vs_n_summary(results, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)

    algorithms = ["Greedy (input order)", "Welsh-Powell", "DSatur"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for alg in algorithms:
        subset = [r for r in results if r["algorithm"] == alg]
        subset = sorted(subset, key=lambda r: r["n_vertices"])
        n_values = [r["n_vertices"] for r in subset]
        colors = [r["colors"] for r in subset]

        ax.plot(
            n_values,
            colors,
            marker="o",
            linestyle="-",
            label=alg,
            color=ALG_COLOR_MAP.get(alg, None),
        )

    ax.set_xlabel("Number of Vertices (n)")
    ax.set_ylabel("Number of Colors")
    ax.set_title("Colors Used vs Graph Size")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(out_dir, "summary_colors_vs_n.png")
    fig.savefig(out_path, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved Colors vs n summary to {out_path}")


def plot_time_vs_n_summary(results, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)

    algorithms = ["Greedy (input order)", "Welsh-Powell", "DSatur"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for alg in algorithms:
        subset = [r for r in results if r["algorithm"] == alg]
        subset = sorted(subset, key=lambda r: r["n_vertices"])
        n_values = [r["n_vertices"] for r in subset]
        times = [max(r["time_sec"], 1e-5) for r in subset]

        ax.plot(
            n_values,
            times,
            marker="o",
            linestyle="-",
            label=alg,
            color=ALG_COLOR_MAP.get(alg, None),
        )

    ax.set_xlabel("Number of Vertices (n)")
    ax.set_ylabel("Runtime (seconds, log scale)")
    ax.set_yscale("log")
    ax.set_title("Runtime vs Graph Size")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(out_dir, "summary_time_vs_n.png")
    fig.savefig(out_path, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Saved Time vs n summary to {out_path}")


# -----------------------------
# Main entry point
# -----------------------------


def main():
    # Build instances (graphs)
    instances = build_instances()
    # Run experiments (CSV + bar data)
    results = run_experiments()
    # Per-instance horizontal bar plots for colors and runtimes
    plot_all_instances(results)
    # Colored node graphs for small / special / stress instances
    create_colored_graphs(instances)
    # Stress comparison figure (Greedy vs DSatur)
    plot_stress_comparison(instances)
    # Chromatic gap summary (where optimal is known)
    plot_chromatic_gap_summary(results)
    # Chromatic gap summary table + combined figure
    plot_gap_table_and_chart(results)
    # Colors vs n and Time vs n trends
    plot_colors_vs_n_summary(results)
    plot_time_vs_n_summary(results)

    print("All plots generated in 'plots' and 'colored_graphs' directories.")


if __name__ == "__main__":
    main()
