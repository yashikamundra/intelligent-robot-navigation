# Problem 10: Intelligent Robot Navigation (A* Search vs BFS)

An interactive, fully functional, and algorithmically rigorous Python application with a modern Graphical User Interface (GUI) designed for **Problem 10: Intelligent Robot Navigation**.

This application simulates autonomous path planning on 2D occupancy grids, providing real-time visual comparisons between informed heuristic search (**A\***) and uninformed uniform search (**Breadth-First Search / BFS**).

---

## 📸 Key Features

- **Interactive 2D Occupancy Grid**:
  - Dynamically adjustable dimensions (15×15, 20×20, 25×25, 30×30).
  - Left-click and drag to paint walls; right-click and drag to quickly erase.
  - Dedicated tool palette (`Start`, `Goal`, `Wall`, `Eraser`).
  - Direct drag-and-drop repositioning of Start (**S**) and Goal (**G**) nodes.

- **Algorithms Implemented**:
  - **A\* Search (Informed Search)** with admissible heuristics:
    - *Manhattan Distance ($L_1$)*: Optimal for 4-directional grid movement.
    - *Euclidean Distance ($L_2$)*: Straight-line physical distance.
    - *Octile / Diagonal Distance*: Optimal and tightest heuristic for 8-directional movement.
  - **Breadth-First Search (BFS) (Uninformed Search)**:
    - Expands level-by-level uniformly in concentric rings; guarantees shortest path in unweighted grids.

- **Real-Time Step-by-Step Visualization**:
  - Color-coded cells: Unvisited, Obstacles, Open Set (Frontier), Closed Set (Visited/Explored), and Final Shortest Path.
  - Animated robot traversal along the computed trajectory upon reaching the target.
  - Granular animation speed control (from `0 ms` instant benchmark to `60 ms` educational stepping).
  - Play, Pause/Resume, and Single-Step execution modes.

- **Procedural Map Presets**:
  - 🎲 **Random Map Generator** (configurable density).
  - 🌀 **Recursive Division Maze Generator** (creates authentic labyrinth corridors and dead-ends).
  - 🧲 **U-Trap (Convex Local Minima)** (demonstrates dramatic differences when naive heuristics or uninformed search wrap around obstacles).
  - 〰️ **Alternating Corridors** (slalom test).

- **Performance Metrics Dashboard**:
  - **Status**: Target reached or unreachable.
  - **Path Length**: Number of steps from Start to Goal.
  - **Total Path Cost**: Cumulative edge weight ($1.0$ for orthogonal, $\sqrt{2} \approx 1.414$ for diagonals).
  - **Nodes Explored**: Total cells closed during search.
  - **Search Time**: Pure algorithmic latency measured in milliseconds (`ms`) with microsecond resolution.

- **Direct A\* vs BFS Comparison Feature**:
  - Head-to-head benchmarking on the exact same layout.
  - Tabular comparison modal displaying solvability, path length, path cost, nodes expanded, and time delta.
  - Highlights search-space reduction percentage achieved by heuristic guidance (often **70%–85% fewer nodes explored** by A\*).

---

## 🛠️ Project Structure

The project is structured cleanly into decoupled modules and also includes an optional all-in-one standalone script:

```
AI/
├── algorithms.py                  # Pure search algorithms (A*, BFS, heuristics, metrics)
├── grid_model.py                  # 2D occupancy grid, topology, procedural generators
├── gui.py                         # Tkinter modern dark GUI, canvas renderer, telemetry cards
├── robot_navigation.py            # Main modular application launcher
├── robot_navigation_standalone.py # Self-contained single-file distribution
└── README.md                      # Comprehensive documentation and algorithmic analysis
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.8 or higher.
- Built-in `tkinter` (included by default in standard Python installations on Windows, macOS, and Linux).
- **No external `pip` dependencies are required!**

### Running the Modular Application (Recommended)
Open a terminal (Command Prompt, PowerShell, or Bash) in this directory:

```bash
python robot_navigation.py
```

You can optionally specify custom starting dimensions:
```bash
python robot_navigation.py --rows 25 --cols 25
```

### Running the Standalone Single-File Version
If you prefer a single portable script:
```bash
python robot_navigation_standalone.py
```

---

## 🧠 Theoretical Background

### 1. A* Search Evaluation Function
A* evaluates nodes by combining the exact path cost from the start node and an estimated cost to the goal:
$$f(n) = g(n) + h(n)$$
where:
- $g(n)$: Exact accumulated cost from the start node to node $n$.
- $h(n)$: Admissible heuristic estimate from node $n$ to the goal.
- $f(n)$: Estimated total cost of the cheapest solution passing through $n$.

### 2. Admissibility and Optimality
A heuristic $h(n)$ is **admissible** if it never overestimates the true cost to reach the goal:
$$h(n) \le h^*(n), \quad \forall n$$
Since all implemented heuristics (Manhattan, Euclidean, Octile) never exceed the true geodesic distance on the grid, A* is mathematically guaranteed to return the **optimal (shortest) path**.

### 3. Consistency (Monotonicity)
A heuristic is **consistent** if for every node $n$ and each successor $n'$ generated by an action $a$:
$$h(n) \le c(n, a, n') + h(n')$$
The triangular inequality holds for Manhattan, Euclidean, and Octile distance metrics on 2D grids. Therefore, when A* expands a node, the shortest path to that node is already known, ensuring nodes do not need to be reopened.

### 4. Comparison: A* vs BFS
| Metric | Breadth-First Search (BFS) | A* Search (Informed) |
| :--- | :--- | :--- |
| **Strategy** | Uninformed (blind FIFO queue) | Informed (Best-First via Min-Heap Priority Queue) |
| **Frontier Order** | Explores outward in uniform concentric rings | Expands toward the lowest $f(n) = g(n) + h(n)$ |
| **Exploration Space** | High (searches irrelevant directions equally) | Low (focused towards the goal) |
| **Optimality** | Optimal for unweighted graphs | Optimal for arbitrary positive edge costs |
| **Typical Efficiency**| Baseline ($100\%$ nodes) | **70%–90% reduction** in explored nodes |

---

## 🎮 Interactive Controls & Shortcuts

| Action | Control |
| :--- | :--- |
| **Paint Obstacles** | Left-click and drag across grid cells |
| **Erase Obstacles** | Right-click and drag across grid cells (or select Eraser tool) |
| **Move Start / Goal** | Click and drag the `S` or `G` badge directly, or select the tool |
| **Start / Resume Search** | Click `▶ START SEARCH` or press `Spacebar` |
| **Pause Search** | Click `⏸ Pause` |
| **Step Forward** | Click `⏭ Step` (single node expansion) |
| **Compare A* vs BFS** | Click `⚖️ COMPARE A* VS BFS` or press `C` |
| **Clear Visualization** | Click `🔄 Clear Search` or press `Escape` |
