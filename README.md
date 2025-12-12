# Graph Coloring Algorithm Evaluation System
Course Name: CPS 3440 -- Analysis of Algorithms
Members: Qingqing Wu (1306107), Jingwen Zheng (1305944), Qian Wang(1306038), Jing Nie(1306251)


## Project Overview  
This project studies the Graph Coloring Problem, a classical NP-hard problem in combinatorial optimization.
We implement and compare baseline, heuristic, and exact graph coloring algorithms, and conduct systematic empirical evaluations to analyze their trade-offs in solution quality (number of colors) and runtime performance.

The project focuses on answering the following questions:
How poorly can naive greedy coloring perform?
How much improvement can heuristic methods provide in practice?
How close can heuristics get to the optimal solution?
What are the trade-offs between solution quality and computational cost?
##Implemented Algorithms
Greedy Coloring (Input Order) – baseline / control algorithm
Welsh–Powell Algorithm – degree-based heuristic
DSatur Algorithm – saturation-based heuristic
Exact Backtracking Coloring – exact algorithm with branch-and-bound and time limit (for small graphs)
---
## Quick Start  
### Environment Requirements  
- Python 3.8+  
Required Libraries
 matplotlib
 pandas
 numpy
- Install dependencies:  
matplotlib==3.7.1
pandas==1.5.3
numpy==1.23.5
##Running the Project
Run the entire experimental pipeline using:
python main.py
This command will:
1. Generate multiple graph instances
2. Run all coloring algorithms
3. Record experimental results to CSV
4. Generate all comparison plots and visualizations

##Project Structure
Project
├── main.py                          # Entry point for the entire experiment
│
├── graph.py                         # Graph data structure (adjacency list)
├── generators.py                    # Graph instance generators
│   ├── random graphs (Erdos–Renyi)
│   ├── bipartite graphs
│   ├── complete graphs
│   └── stress-test graphs
│
├── algorithms.py                    # Graph coloring algorithms
│   ├── greedy_coloring              # Baseline algorithm
│   ├── welsh_powell_coloring        # Heuristic algorithm
│   ├── dsatur_coloring              # Advanced heuristic
│   └── exact_backtracking_coloring  # Exact algorithm (small graphs only)
│
├── experiments.py                   # Experiment runner and evaluation logic
│
├── plots/                           # Generated comparison plots
│   ├── colors_vs_n.png
│   ├── runtime_vs_n.png
│   ├── chromatic_gap_summary.png
│   └── stress_comparison.png
│
├── colored_graphs/                  # Colored graph visualizations
│
└── results_graph_coloring.csv       # Experimental results (CSV)

##Experimental Design
###Graph Instances
We evaluate algorithms on diverse graph categories:
Small random graphs (exact solution feasible)
Medium random graphs (sparse and dense)
Bipartite graphs (known optimal chromatic number = 2)
Complete graphs (known optimal chromatic number = n)
Stress-test graphs designed to expose weaknesses of greedy coloring

Evaluation Metrics
Number of colors used (solution quality)
Runtime (seconds)
Validity of coloring
Chromatic gap (difference between heuristic and optimal solution when available)

##Key Results and Observations
Greedy coloring is fast but often uses significantly more colors.
Welsh–Powell improves solution quality with minimal runtime overhead.
DSatur consistently produces near-optimal or optimal solutions, especially on stress-test graphs.
Exact coloring confirms optimality on small instances and serves as a benchmark.
There exists a clear trade-off between solution quality and runtime, with DSatur providing the best balance in practice.
##Output Files
###CSV Results
-results_graph_coloring.csv – records colors used, runtime, and gap to optimal.
###Plots
Colors Used vs Graph Size
Runtime vs Graph Size (log scale)
Chromatic Gap Summary
Stress-test comparison (Greedy vs DSatur)
###Graph Visualizations
Colored node plots illustrating qualitative differences between algorithms.

##Limitations and Future Work
Exact algorithms are limited to small graphs due to exponential complexity.
Future improvements may include:
      - Randomized greedy variants
      - Multi-start heuristics
      - Hybrid approaches combining DSatur with local search
      - Evaluation on real-world benchmark datasets (e.g., OR-Library)

Contact

Qingqing Wu: wuqin@kean.edu

Qian Wang: wanqian@kean.edu

Jingwen Zheng: zhejingw@kean.edu

Jing Nie: nieji@kean.edu




