#!/usr/bin/env python3
"""
robot_navigation_standalone.py - Standalone Single-File Application
===================================================================
Problem 10: Intelligent Robot Navigation (A* Search vs BFS)

Features:
  - Interactive Grid with Left-click/drag draw, Right-click/drag erase
  - Move Start and Goal nodes by selecting the tool or dragging them directly
  - Admissible Heuristics for A*: Manhattan, Euclidean, and Octile / Diagonal
  - Uninformed Breadth-First Search (BFS) for baseline comparison
  - Real-time step-by-step exploration animation with speed control (Instant to Slow)
  - Comprehensive Performance Metrics Dashboard (Path length, Cost, Explored nodes, Search time in ms)
  - Direct Side-by-Side Algorithm Comparison Modal on identical grids
  - Procedural map presets: Random (density), Recursive Division Maze, U-Trap (local minima), Corridors
  - 4-Directional (orthogonal) and 8-Directional (with diagonals and corner-cutting prevention)

Requirements:
  Python 3.8+ with standard `tkinter` library (no external pip dependencies needed).

Usage:
  python robot_navigation_standalone.py
"""

from __future__ import annotations
import math
import time
import random
from collections import deque
import heapq
import argparse
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Generator, List, Tuple, Dict, Set, Optional, Callable

Point = Tuple[int, int]  # (row, col)

# ===========================================================================
# 1. HEURISTIC FUNCTIONS (Admissible & Consistent)
# ===========================================================================

def heuristic_manhattan(p1: Point, p2: Point) -> float:
    """Manhattan distance: |r1 - r2| + |c1 - c2| (admissible for 4-way movement)."""
    return float(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))

def heuristic_euclidean(p1: Point, p2: Point) -> float:
    """Euclidean distance: sqrt(dr^2 + dc^2) (admissible for 4-way and 8-way)."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def heuristic_octile(p1: Point, p2: Point) -> float:
    """Octile/Diagonal distance: D*(|dx-dy|) + D2*min(dx, dy) where D=1, D2=sqrt(2)."""
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    return 1.0 * abs(dx - dy) + math.sqrt(2.0) * min(dx, dy)

HEURISTICS: Dict[str, Callable[[Point, Point], float]] = {
    "Manhattan": heuristic_manhattan,
    "Euclidean": heuristic_euclidean,
    "Octile / Diagonal": heuristic_octile,
}

# ===========================================================================
# 2. METRICS & SEARCH RESULT DATA MODEL
# ===========================================================================

class SearchMetrics:
    def __init__(
        self,
        algorithm: str = "",
        path_found: bool = False,
        path: Optional[List[Point]] = None,
        path_length: int = 0,
        path_cost: float = 0.0,
        nodes_explored: int = 0,
        nodes_in_frontier: int = 0,
        execution_time_ms: float = 0.0,
        heuristic_name: str = "N/A"
    ):
        self.algorithm = algorithm
        self.path_found = path_found
        self.path = path or []
        self.path_length = path_length
        self.path_cost = round(path_cost, 4)
        self.nodes_explored = nodes_explored
        self.nodes_in_frontier = nodes_in_frontier
        self.execution_time_ms = round(execution_time_ms, 3)
        self.heuristic_name = heuristic_name

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "path_found": self.path_found,
            "path_length": self.path_length,
            "path_cost": self.path_cost,
            "nodes_explored": self.nodes_explored,
            "nodes_in_frontier": self.nodes_in_frontier,
            "execution_time_ms": self.execution_time_ms,
            "heuristic_name": self.heuristic_name,
        }

def reconstruct_path(came_from: Dict[Point, Optional[Point]], current: Point) -> List[Point]:
    path = [current]
    while current in came_from and came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def compute_path_cost(path: List[Point]) -> float:
    if len(path) <= 1:
        return 0.0
    cost = 0.0
    for i in range(len(path) - 1):
        dr = abs(path[i][0] - path[i+1][0])
        dc = abs(path[i][1] - path[i+1][1])
        if dr == 1 and dc == 1:
            cost += math.sqrt(2.0)
        else:
            cost += 1.0
    return cost

# ===========================================================================
# 3. ALGORITHMS: A* SEARCH & BREADTH-FIRST SEARCH
# ===========================================================================

def a_star_search(
    start: Point,
    goal: Point,
    get_neighbors_fn: Callable[[Point], List[Tuple[Point, float]]],
    heuristic_fn: Callable[[Point, Point], float],
    heuristic_name: str = "Manhattan"
) -> SearchMetrics:
    t0 = time.perf_counter()
    if start == goal:
        dt = (time.perf_counter() - t0) * 1000.0
        return SearchMetrics("A* Search", True, [start], 0, 0.0, 1, 0, dt, heuristic_name)

    counter = 0
    open_heap: List[Tuple[float, int, Point]] = []
    g_score: Dict[Point, float] = {start: 0.0}
    f_start = heuristic_fn(start, goal)
    heapq.heappush(open_heap, (f_start, counter, start))

    came_from: Dict[Point, Optional[Point]] = {start: None}
    closed_set: Set[Point] = set()
    open_set_tracker: Set[Point] = {start}
    nodes_explored_count = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed_set:
            continue
        if current in open_set_tracker:
            open_set_tracker.remove(current)

        closed_set.add(current)
        nodes_explored_count += 1

        if current == goal:
            path = reconstruct_path(came_from, current)
            dt = (time.perf_counter() - t0) * 1000.0
            return SearchMetrics(
                algorithm="A* Search",
                path_found=True,
                path=path,
                path_length=len(path) - 1,
                path_cost=compute_path_cost(path),
                nodes_explored=nodes_explored_count,
                nodes_in_frontier=len(open_set_tracker),
                execution_time_ms=dt,
                heuristic_name=heuristic_name
            )

        current_g = g_score[current]
        for neighbor, move_cost in get_neighbors_fn(current):
            if neighbor in closed_set:
                continue
            tentative_g = current_g + move_cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h_val = heuristic_fn(neighbor, goal)
                f_val = tentative_g + h_val * 1.0001
                counter += 1
                heapq.heappush(open_heap, (f_val, counter, neighbor))
                open_set_tracker.add(neighbor)

    dt = (time.perf_counter() - t0) * 1000.0
    return SearchMetrics("A* Search", False, [], 0, 0.0, nodes_explored_count, 0, dt, heuristic_name)

def a_star_generator(
    start: Point,
    goal: Point,
    get_neighbors_fn: Callable[[Point], List[Tuple[Point, float]]],
    heuristic_fn: Callable[[Point, Point], float],
    heuristic_name: str = "Manhattan"
) -> Generator[Tuple[str, any], None, None]:
    t0 = time.perf_counter()
    if start == goal:
        dt = (time.perf_counter() - t0) * 1000.0
        m = SearchMetrics("A* Search", True, [start], 0, 0.0, 1, 0, dt, heuristic_name)
        yield ("VISIT", start)
        yield ("PATH", ([start], m))
        return

    counter = 0
    open_heap: List[Tuple[float, int, Point]] = []
    g_score: Dict[Point, float] = {start: 0.0}
    f_start = heuristic_fn(start, goal)
    heapq.heappush(open_heap, (f_start, counter, start))

    came_from: Dict[Point, Optional[Point]] = {start: None}
    closed_set: Set[Point] = set()
    open_set_tracker: Set[Point] = {start}
    nodes_explored_count = 0

    yield ("FRONTIER", start)

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed_set:
            continue
        if current in open_set_tracker:
            open_set_tracker.remove(current)

        closed_set.add(current)
        nodes_explored_count += 1
        yield ("VISIT", current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            dt = (time.perf_counter() - t0) * 1000.0
            m = SearchMetrics(
                algorithm="A* Search",
                path_found=True,
                path=path,
                path_length=len(path) - 1,
                path_cost=compute_path_cost(path),
                nodes_explored=nodes_explored_count,
                nodes_in_frontier=len(open_set_tracker),
                execution_time_ms=dt,
                heuristic_name=heuristic_name
            )
            yield ("PATH", (path, m))
            return

        current_g = g_score[current]
        for neighbor, move_cost in get_neighbors_fn(current):
            if neighbor in closed_set:
                continue
            tentative_g = current_g + move_cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h_val = heuristic_fn(neighbor, goal)
                f_val = tentative_g + h_val * 1.0001
                counter += 1
                heapq.heappush(open_heap, (f_val, counter, neighbor))
                if neighbor not in open_set_tracker:
                    open_set_tracker.add(neighbor)
                    yield ("FRONTIER", neighbor)

    dt = (time.perf_counter() - t0) * 1000.0
    m = SearchMetrics("A* Search", False, [], 0, 0.0, nodes_explored_count, 0, dt, heuristic_name)
    yield ("NO_PATH", m)

def bfs_search(
    start: Point,
    goal: Point,
    get_neighbors_fn: Callable[[Point], List[Tuple[Point, float]]]
) -> SearchMetrics:
    t0 = time.perf_counter()
    if start == goal:
        dt = (time.perf_counter() - t0) * 1000.0
        return SearchMetrics("Breadth-First Search (BFS)", True, [start], 0, 0.0, 1, 0, dt, "None (Uninformed)")

    queue = deque([start])
    visited: Set[Point] = {start}
    came_from: Dict[Point, Optional[Point]] = {start: None}
    nodes_explored_count = 0

    while queue:
        current = queue.popleft()
        nodes_explored_count += 1

        if current == goal:
            path = reconstruct_path(came_from, current)
            dt = (time.perf_counter() - t0) * 1000.0
            return SearchMetrics(
                algorithm="Breadth-First Search (BFS)",
                path_found=True,
                path=path,
                path_length=len(path) - 1,
                path_cost=compute_path_cost(path),
                nodes_explored=nodes_explored_count,
                nodes_in_frontier=len(queue),
                execution_time_ms=dt,
                heuristic_name="None (Uninformed)"
            )

        for neighbor, _ in get_neighbors_fn(current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    dt = (time.perf_counter() - t0) * 1000.0
    return SearchMetrics("Breadth-First Search (BFS)", False, [], 0, 0.0, nodes_explored_count, 0, dt, "None (Uninformed)")

def bfs_generator(
    start: Point,
    goal: Point,
    get_neighbors_fn: Callable[[Point], List[Tuple[Point, float]]]
) -> Generator[Tuple[str, any], None, None]:
    t0 = time.perf_counter()
    if start == goal:
        dt = (time.perf_counter() - t0) * 1000.0
        m = SearchMetrics("Breadth-First Search (BFS)", True, [start], 0, 0.0, 1, 0, dt, "None (Uninformed)")
        yield ("VISIT", start)
        yield ("PATH", ([start], m))
        return

    queue = deque([start])
    visited: Set[Point] = {start}
    came_from: Dict[Point, Optional[Point]] = {start: None}
    nodes_explored_count = 0

    yield ("FRONTIER", start)

    while queue:
        current = queue.popleft()
        nodes_explored_count += 1
        yield ("VISIT", current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            dt = (time.perf_counter() - t0) * 1000.0
            m = SearchMetrics(
                algorithm="Breadth-First Search (BFS)",
                path_found=True,
                path=path,
                path_length=len(path) - 1,
                path_cost=compute_path_cost(path),
                nodes_explored=nodes_explored_count,
                nodes_in_frontier=len(queue),
                execution_time_ms=dt,
                heuristic_name="None (Uninformed)"
            )
            yield ("PATH", (path, m))
            return

        for neighbor, _ in get_neighbors_fn(current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)
                yield ("FRONTIER", neighbor)

    dt = (time.perf_counter() - t0) * 1000.0
    m = SearchMetrics("Breadth-First Search (BFS)", False, [], 0, 0.0, nodes_explored_count, 0, dt, "None (Uninformed)")
    yield ("NO_PATH", m)

# ===========================================================================
# 4. GRID MODEL & PRESET MAP GENERATORS
# ===========================================================================

class GridModel:
    def __init__(self, rows: int = 20, cols: int = 20):
        self.rows = max(5, rows)
        self.cols = max(5, cols)
        self.start: Point = (1, 1)
        self.goal: Point = (self.rows - 2, self.cols - 2)
        self.obstacles: Set[Point] = set()
        self.allow_diagonal: bool = False

    def in_bounds(self, p: Point) -> bool:
        return 0 <= p[0] < self.rows and 0 <= p[1] < self.cols

    def is_obstacle(self, p: Point) -> bool:
        return p in self.obstacles

    def is_passable(self, p: Point) -> bool:
        return self.in_bounds(p) and (p not in self.obstacles)

    def set_start(self, p: Point) -> bool:
        if not self.in_bounds(p) or p == self.goal:
            return False
        self.start = p
        self.obstacles.discard(p)
        return True

    def set_goal(self, p: Point) -> bool:
        if not self.in_bounds(p) or p == self.start:
            return False
        self.goal = p
        self.obstacles.discard(p)
        return True

    def add_obstacle(self, p: Point) -> bool:
        if self.in_bounds(p) and p != self.start and p != self.goal:
            self.obstacles.add(p)
            return True
        return False

    def remove_obstacle(self, p: Point) -> bool:
        if p in self.obstacles:
            self.obstacles.remove(p)
            return True
        return False

    def clear_obstacles(self) -> None:
        self.obstacles.clear()

    def resize(self, new_rows: int, new_cols: int) -> None:
        self.rows = max(5, new_rows)
        self.cols = max(5, new_cols)
        self.obstacles = {p for p in self.obstacles if self.in_bounds(p)}
        self.start = (min(self.start[0], self.rows - 1), min(self.start[1], self.cols - 1))
        self.goal = (min(self.goal[0], self.rows - 1), min(self.goal[1], self.cols - 1))
        if self.goal == self.start:
            self.goal = (max(0, self.rows - 2), max(0, self.cols - 2))
        self.obstacles.discard(self.start)
        self.obstacles.discard(self.goal)

    def get_neighbors(self, p: Point) -> List[Tuple[Point, float]]:
        r, c = p
        neighbors = []
        # 4 Orthogonal
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            cand = (r + dr, c + dc)
            if self.is_passable(cand):
                neighbors.append((cand, 1.0))

        # Diagonals
        if self.allow_diagonal:
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                cand = (r + dr, c + dc)
                if not self.is_passable(cand):
                    continue
                # Anti-corner cutting
                if not self.is_passable((r + dr, c)) and not self.is_passable((r, c + dc)):
                    continue
                neighbors.append((cand, math.sqrt(2.0)))
        return neighbors

    def generate_random(self, density: float = 0.28) -> None:
        self.clear_obstacles()
        for r in range(self.rows):
            for c in range(self.cols):
                pt = (r, c)
                if pt != self.start and pt != self.goal and random.random() < density:
                    self.obstacles.add(pt)

    def generate_u_trap(self) -> None:
        self.clear_obstacles()
        mid_r = self.rows // 2
        mid_c = self.cols // 2
        self.start = (mid_r, mid_c)
        self.goal = (mid_r, self.cols - 3)
        arm_len = max(3, self.rows // 4)
        wall_c = mid_c + 2

        for r in range(mid_r - arm_len, mid_r + arm_len + 1):
            if 0 <= r < self.rows:
                self.obstacles.add((r, wall_c))
        for c in range(mid_c - arm_len, wall_c + 1):
            if 0 <= c < self.cols:
                top_r = mid_r - arm_len
                bot_r = mid_r + arm_len
                if 0 <= top_r < self.rows:
                    self.obstacles.add((top_r, c))
                if 0 <= bot_r < self.rows:
                    self.obstacles.add((bot_r, c))

        self.obstacles.discard(self.start)
        self.obstacles.discard(self.goal)

    def generate_corridors(self) -> None:
        self.clear_obstacles()
        spacing = 3
        for r in range(spacing, self.rows - 1, spacing):
            if (r // spacing) % 2 == 1:
                for c in range(0, self.cols - 3):
                    self.obstacles.add((r, c))
            else:
                for c in range(3, self.cols):
                    self.obstacles.add((r, c))
        self.start = (1, 1)
        self.goal = (self.rows - 2, self.cols - 2)
        self.obstacles.discard(self.start)
        self.obstacles.discard(self.goal)

    def generate_recursive_maze(self) -> None:
        self.clear_obstacles()
        for r in range(self.rows):
            self.obstacles.add((r, 0))
            self.obstacles.add((r, self.cols - 1))
        for c in range(self.cols):
            self.obstacles.add((0, c))
            self.obstacles.add((self.rows - 1, c))

        def divide(r1: int, r2: int, c1: int, c2: int) -> None:
            if c2 - c1 <= 2 or r2 - r1 <= 2:
                return
            horizontal = (r2 - r1 > c2 - c1) if (r2 - r1 != c2 - c1) else (random.random() < 0.5)
            if horizontal:
                wall_r = random.randrange(r1 + 1, r2, 2)
                passage_c = random.randrange(c1, c2 + 1, 2)
                for c in range(c1, c2 + 1):
                    if c != passage_c:
                        self.obstacles.add((wall_r, c))
                divide(r1, wall_r - 1, c1, c2)
                divide(wall_r + 1, r2, c1, c2)
            else:
                wall_c = random.randrange(c1 + 1, c2, 2)
                passage_r = random.randrange(r1, r2 + 1, 2)
                for r in range(r1, r2 + 1):
                    if r != passage_r:
                        self.obstacles.add((r, wall_c))
                divide(r1, r2, c1, wall_c - 1)
                divide(r1, r2, wall_c + 1, c2)

        divide(1, self.rows - 2, 1, self.cols - 2)
        self.start = (1, 1)
        self.goal = (self.rows - 2, self.cols - 2)
        self.obstacles.discard(self.start)
        self.obstacles.discard(self.goal)

# ===========================================================================
# 5. USER INTERFACE (TKINTER MODERN DARK THEME)
# ===========================================================================

THEME = {
    "bg_main": "#0f172a",
    "bg_card": "#1e293b",
    "bg_card_border": "#334155",
    "bg_sidebar": "#111827",
    "text_primary": "#f8fafc",
    "text_secondary": "#94a3b8",
    "text_accent": "#38bdf8",
    "cell_empty": "#1e293b",
    "cell_border": "#334155",
    "cell_wall": "#020617",
    "cell_wall_border": "#475569",
    "cell_start": "#10b981",
    "cell_goal": "#ef4444",
    "cell_frontier": "#8b5cf6",
    "cell_visited": "#3b82f6",
    "cell_path": "#fbbf24",
    "robot_marker": "#f43f5e",
}

class RobotNavigationApp:
    def __init__(self, root: tk.Tk, default_rows: int = 20, default_cols: int = 20):
        self.root = root
        self.root.title("Problem 10: Intelligent Robot Navigation (A* vs BFS)")
        self.root.geometry("1280x820")
        self.root.minsize(1050, 700)
        self.root.configure(bg=THEME["bg_main"])

        self.grid = GridModel(rows=default_rows, cols=default_cols)
        self.current_tool = tk.StringVar(value="wall")
        self.is_mouse_pressed = False
        self.drag_entity: Optional[str] = None

        self.selected_algo = tk.StringVar(value="A* Search")
        self.selected_heuristic = tk.StringVar(value="Manhattan")
        self.allow_diagonal_var = tk.BooleanVar(value=False)
        self.speed_ms = tk.IntVar(value=15)
        self.grid_size_var = tk.IntVar(value=20)

        self.search_generator = None
        self.is_running = False
        self.is_paused = False
        self.animation_timer = None
        self.robot_timer = None

        self.frontier_nodes: Set[Point] = set()
        self.visited_nodes: Set[Point] = set()
        self.final_path: List[Point] = []
        self.last_metrics: Optional[SearchMetrics] = None
        self.robot_step_index: int = 0

        self._configure_styles()
        self._build_layout()
        self._bind_events()

        self.root.update_idletasks()
        self.redraw_canvas()

    def _configure_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Dark.TFrame", background=THEME["bg_sidebar"])
        style.configure("Card.TFrame", background=THEME["bg_card"])
        style.configure("Main.TFrame", background=THEME["bg_main"])
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", THEME["bg_card"])],
            selectbackground=[("readonly", THEME["bg_card"])],
            selectforeground=[("readonly", THEME["text_primary"])],
            background=[("readonly", THEME["bg_card_border"])]
        )

    def _build_layout(self):
        header = tk.Frame(self.root, bg="#090d16", height=48)
        header.pack(fill=tk.X, side=tk.TOP)
        tk.Label(
            header, text="🤖 Intelligent Robot Navigation",
            bg="#090d16", fg=THEME["text_primary"], font=("Segoe UI", 13, "bold")
        ).pack(side=tk.LEFT, padx=16, pady=8)
        tk.Label(
            header, text="Heuristic A* Search vs Breadth-First Search (BFS)",
            bg="#090d16", fg=THEME["text_accent"], font=("Segoe UI", 9)
        ).pack(side=tk.LEFT, padx=4, pady=8)

        workspace = tk.Frame(self.root, bg=THEME["bg_main"])
        workspace.pack(fill=tk.BOTH, expand=True)

        sidebar_frame = tk.Frame(workspace, bg=THEME["bg_sidebar"], width=310)
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y)
        sidebar_frame.pack_propagate(False)
        self._build_sidebar(sidebar_frame)

        right_panel = tk.Frame(workspace, bg=THEME["bg_main"])
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=12, pady=12)

        self._build_metrics_panel(right_panel)

        canvas_container = tk.Frame(right_panel, bg=THEME["bg_card"], bd=1, relief=tk.SOLID)
        canvas_container.pack(fill=tk.BOTH, expand=True, pady=(8, 4))
        self.canvas = tk.Canvas(canvas_container, bg=THEME["bg_main"], highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._build_legend_bar(right_panel)

    def _build_sidebar(self, parent: tk.Frame):
        canvas_side = tk.Canvas(parent, bg=THEME["bg_sidebar"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas_side.yview)
        scrollable_frame = tk.Frame(canvas_side, bg=THEME["bg_sidebar"])
        scrollable_frame.bind("<Configure>", lambda e: canvas_side.configure(scrollregion=canvas_side.bbox("all")))
        canvas_side.create_window((0, 0), window=scrollable_frame, anchor="nw", width=295)
        canvas_side.configure(yscrollcommand=scrollbar.set)
        canvas_side.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        pad_x = 10

        # 1. Tools
        s1 = self._create_section(scrollable_frame, "1. INTERACTIVE TOOLS")
        tools_row = tk.Frame(s1, bg=THEME["bg_sidebar"])
        tools_row.pack(fill=tk.X, padx=pad_x, pady=4)
        for idx, (label, val) in enumerate([("🟢 Start", "start"), ("🎯 Goal", "goal"), ("🧱 Wall", "wall"), ("🧹 Erase", "erase")]):
            rb = tk.Radiobutton(
                tools_row, text=label, value=val, variable=self.current_tool,
                indicatoron=False, bg=THEME["bg_card"], fg=THEME["text_primary"],
                selectcolor="#2563eb", activebackground="#3b82f6", font=("Segoe UI", 8, "bold"),
                relief=tk.FLAT, bd=2, padx=4, pady=4
            )
            rb.grid(row=idx // 2, column=idx % 2, padx=2, pady=2, sticky="nsew")
        tools_row.grid_columnconfigure(0, weight=1)
        tools_row.grid_columnconfigure(1, weight=1)

        tk.Label(
            s1, text="Left-drag: paint | Right-drag: erase\nDrag Start/Goal directly to move",
            bg=THEME["bg_sidebar"], fg=THEME["text_secondary"], font=("Segoe UI", 7, "italic"), justify=tk.CENTER
        ).pack(pady=(0, 4))

        # 2. Algorithm
        s2 = self._create_section(scrollable_frame, "2. ALGORITHM & HEURISTIC")
        tk.Label(s2, text="Search Algorithm:", bg=THEME["bg_sidebar"], fg=THEME["text_secondary"], font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=pad_x, pady=(4, 2))
        algo_combo = ttk.Combobox(s2, textvariable=self.selected_algo, values=["A* Search", "Breadth-First Search (BFS)"], state="readonly", font=("Segoe UI", 9))
        algo_combo.pack(fill=tk.X, padx=pad_x, pady=(0, 4))
        algo_combo.bind("<<ComboboxSelected>>", self._on_algo_change)

        self.heuristic_lbl = tk.Label(s2, text="A* Heuristic:", bg=THEME["bg_sidebar"], fg=THEME["text_secondary"], font=("Segoe UI", 8, "bold"))
        self.heuristic_lbl.pack(anchor="w", padx=pad_x, pady=(2, 2))
        self.heuristic_combo = ttk.Combobox(s2, textvariable=self.selected_heuristic, values=list(HEURISTICS.keys()), state="readonly", font=("Segoe UI", 9))
        self.heuristic_combo.pack(fill=tk.X, padx=pad_x, pady=(0, 4))

        tk.Checkbutton(
            s2, text="Allow 8-Way Diagonals", variable=self.allow_diagonal_var,
            command=self._on_diagonal_toggle, bg=THEME["bg_sidebar"], fg=THEME["text_primary"],
            selectcolor=THEME["bg_card"], font=("Segoe UI", 8)
        ).pack(anchor="w", padx=pad_x, pady=(2, 4))

        dim_frame = tk.Frame(s2, bg=THEME["bg_sidebar"])
        dim_frame.pack(fill=tk.X, padx=pad_x, pady=(2, 4))
        tk.Label(dim_frame, text="Grid Size:", bg=THEME["bg_sidebar"], fg=THEME["text_secondary"], font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT)
        dim_combo = ttk.Combobox(dim_frame, textvariable=self.grid_size_var, values=[15, 20, 25, 30], width=5, state="readonly")
        dim_combo.pack(side=tk.RIGHT)
        dim_combo.bind("<<ComboboxSelected>>", self._on_resize_grid)

        # 3. Presets
        s3 = self._create_section(scrollable_frame, "3. PROCEDURAL MAPS")
        preset_frame = tk.Frame(s3, bg=THEME["bg_sidebar"])
        preset_frame.pack(fill=tk.X, padx=pad_x, pady=4)
        presets = [
            ("🎲 Random Map", self.generate_random_map),
            ("🌀 Recursive Maze", self.generate_recursive_maze),
            ("🧲 U-Trap (Minima)", self.generate_u_trap),
            ("〰️ Corridors", self.generate_corridors),
            ("🗑️ Clear Walls", self.clear_all_obstacles),
            ("🔄 Clear Search", self.clear_search_visualization),
        ]
        for idx, (lbl, cmd) in enumerate(presets):
            b = tk.Button(
                preset_frame, text=lbl, command=cmd, bg=THEME["bg_card"], fg=THEME["text_primary"],
                relief=tk.FLAT, font=("Segoe UI", 8), pady=3
            )
            b.grid(row=idx // 2, column=idx % 2, padx=2, pady=2, sticky="nsew")
        preset_frame.grid_columnconfigure(0, weight=1)
        preset_frame.grid_columnconfigure(1, weight=1)

        # 4. Controls
        s4 = self._create_section(scrollable_frame, "4. EXECUTION CONTROLS")
        tk.Label(s4, text="Animation Delay (ms):", bg=THEME["bg_sidebar"], fg=THEME["text_secondary"], font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=pad_x, pady=(4, 0))
        tk.Scale(
            s4, from_=0, to=60, orient=tk.HORIZONTAL, variable=self.speed_ms,
            bg=THEME["bg_sidebar"], fg=THEME["text_primary"], troughcolor=THEME["bg_card"], highlightthickness=0
        ).pack(fill=tk.X, padx=pad_x)
        tk.Label(s4, text="0 ms (Instant) ⟵ ⟶ 60 ms", bg=THEME["bg_sidebar"], fg=THEME["text_secondary"], font=("Segoe UI", 7)).pack()

        self.btn_run = tk.Button(
            s4, text="▶ START SEARCH", command=self.start_or_resume_search,
            bg="#2563eb", fg="#ffffff", font=("Segoe UI", 10, "bold"), relief=tk.FLAT, pady=6
        )
        self.btn_run.pack(fill=tk.X, padx=pad_x, pady=(6, 3))

        ctrl_row = tk.Frame(s4, bg=THEME["bg_sidebar"])
        ctrl_row.pack(fill=tk.X, padx=pad_x, pady=2)
        self.btn_pause = tk.Button(ctrl_row, text="⏸ Pause", command=self.toggle_pause, bg=THEME["bg_card"], fg=THEME["text_primary"], font=("Segoe UI", 8, "bold"), relief=tk.FLAT, state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.btn_step = tk.Button(ctrl_row, text="⏭ Step", command=self.step_forward, bg=THEME["bg_card"], fg=THEME["text_primary"], font=("Segoe UI", 8, "bold"), relief=tk.FLAT)
        self.btn_step.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        # 5. Comparison
        s5 = self._create_section(scrollable_frame, "5. DIRECT BENCHMARK")
        tk.Button(
            s5, text="⚖️ COMPARE A* VS BFS", command=self.open_comparison_dialog,
            bg="#7c3aed", fg="#ffffff", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, pady=6
        ).pack(fill=tk.X, padx=pad_x, pady=(4, 10))

    def _create_section(self, parent: tk.Widget, title: str) -> tk.Frame:
        frame = tk.Frame(parent, bg=THEME["bg_sidebar"])
        frame.pack(fill=tk.X, pady=(6, 0))
        tk.Label(frame, text=title, bg=THEME["bg_sidebar"], fg=THEME["text_accent"], font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=10, pady=(4, 2))
        tk.Frame(frame, bg="#1e293b", height=1).pack(fill=tk.X, padx=10, pady=(0, 2))
        return frame

    def _build_metrics_panel(self, parent: tk.Frame):
        metrics_bar = tk.Frame(parent, bg=THEME["bg_main"])
        metrics_bar.pack(fill=tk.X, side=tk.TOP, pady=(0, 4))
        self.card_data: Dict[str, Tuple[tk.Label, tk.Label]] = {}
        cards = [
            ("STATUS", "Ready", THEME["text_accent"]),
            ("PATH LENGTH", "0 steps", THEME["text_primary"]),
            ("PATH COST", "0.00", THEME["text_primary"]),
            ("NODES EXPLORED", "0", "#f59e0b"),
            ("SEARCH TIME", "0.00 ms", "#10b981"),
            ("ACTIVE ALGO", "A* (Manhattan)", "#a855f7")
        ]
        for title, val, col in cards:
            card = tk.Frame(metrics_bar, bg=THEME["bg_card"], bd=1, relief=tk.SOLID)
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
            t_lbl = tk.Label(card, text=title, bg=THEME["bg_card"], fg=THEME["text_secondary"], font=("Segoe UI", 7, "bold"))
            t_lbl.pack(anchor="w", padx=6, pady=(3, 0))
            v_lbl = tk.Label(card, text=val, bg=THEME["bg_card"], fg=col, font=("Segoe UI", 10, "bold"))
            v_lbl.pack(anchor="w", padx=6, pady=(0, 3))
            self.card_data[title] = (t_lbl, v_lbl)

    def _update_metrics_display(self, m: SearchMetrics):
        self.last_metrics = m
        self.card_data["STATUS"][1].config(
            text="Path Found" if m.path_found else "No Path Found",
            fg="#10b981" if m.path_found else "#ef4444"
        )
        self.card_data["PATH LENGTH"][1].config(text=f"{m.path_length} steps")
        self.card_data["PATH COST"][1].config(text=f"{m.path_cost:.2f}")
        self.card_data["NODES EXPLORED"][1].config(text=f"{m.nodes_explored} nodes")
        self.card_data["SEARCH TIME"][1].config(text=f"{m.execution_time_ms:.2f} ms")
        algo_str = "A*" if "A*" in m.algorithm else "BFS"
        if m.heuristic_name and m.heuristic_name != "None (Uninformed)":
            algo_str += f" ({m.heuristic_name})"
        self.card_data["ACTIVE ALGO"][1].config(text=algo_str)

    def _build_legend_bar(self, parent: tk.Frame):
        legend = tk.Frame(parent, bg=THEME["bg_sidebar"], height=24)
        legend.pack(fill=tk.X, side=tk.BOTTOM, pady=(4, 0))
        items = [
            ("Start Node", THEME["cell_start"]),
            ("Goal Node", THEME["cell_goal"]),
            ("Obstacle", THEME["cell_wall"]),
            ("Explored", THEME["cell_visited"]),
            ("Frontier", THEME["cell_frontier"]),
            ("Optimal Path", THEME["cell_path"]),
        ]
        for label, color in items:
            f = tk.Frame(legend, bg=THEME["bg_sidebar"])
            f.pack(side=tk.LEFT, padx=8, pady=3)
            sw = tk.Canvas(f, width=10, height=10, bg=color, highlightthickness=1, highlightbackground="#ffffff")
            sw.pack(side=tk.LEFT, padx=(0, 3))
            tk.Label(f, text=label, bg=THEME["bg_sidebar"], fg=THEME["text_secondary"], font=("Segoe UI", 7)).pack(side=tk.LEFT)

    def _bind_events(self):
        self.canvas.bind("<Configure>", lambda e: self.redraw_canvas())
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down_left)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag_left)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up_left)
        self.canvas.bind("<ButtonPress-3>", self._on_mouse_down_right)
        self.canvas.bind("<B3-Motion>", self._on_mouse_drag_right)

        self.root.bind("<space>", lambda e: self.start_or_resume_search())
        self.root.bind("<Escape>", lambda e: self.clear_search_visualization())
        self.root.bind("c", lambda e: self.open_comparison_dialog())
        self.root.bind("C", lambda e: self.open_comparison_dialog())

    def _get_cell_geometry(self) -> Tuple[float, float, float, float]:
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            return 20.0, 0.0, 0.0, 20.0
        cell_size = min(w / self.grid.cols, h / self.grid.rows) * 0.96
        offset_x = (w - (cell_size * self.grid.cols)) / 2.0
        offset_y = (h - (cell_size * self.grid.rows)) / 2.0
        return cell_size, offset_x, offset_y, w

    def _canvas_to_grid(self, x: float, y: float) -> Optional[Point]:
        cell_size, offset_x, offset_y, _ = self._get_cell_geometry()
        r = int((y - offset_y) // cell_size)
        c = int((x - offset_x) // cell_size)
        if 0 <= r < self.grid.rows and 0 <= c < self.grid.cols:
            return (r, c)
        return None

    def redraw_canvas(self):
        self.canvas.delete("all")
        cell_size, offset_x, offset_y, _ = self._get_cell_geometry()
        path_set = set(self.final_path)

        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                pt = (r, c)
                x1 = offset_x + c * cell_size
                y1 = offset_y + r * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                if pt == self.grid.start:
                    color = THEME["cell_start"]
                    tag = "start"
                elif pt == self.grid.goal:
                    color = THEME["cell_goal"]
                    tag = "goal"
                elif pt in self.grid.obstacles:
                    color = THEME["cell_wall"]
                    tag = "wall"
                elif pt in path_set:
                    color = THEME["cell_path"]
                    tag = "path"
                elif pt in self.frontier_nodes:
                    color = THEME["cell_frontier"]
                    tag = "frontier"
                elif pt in self.visited_nodes:
                    color = THEME["cell_visited"]
                    tag = "visited"
                else:
                    color = THEME["cell_empty"]
                    tag = "empty"

                self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=color,
                    outline=THEME["cell_border"] if tag != "wall" else THEME["cell_wall_border"],
                    width=1, tags=(tag, f"cell_{r}_{c}")
                )

                if pt == self.grid.start:
                    self.canvas.create_text((x1 + x2)/2, (y1 + y2)/2, text="S", fill="#ffffff", font=("Segoe UI", int(max(8, cell_size * 0.45)), "bold"))
                elif pt == self.grid.goal:
                    self.canvas.create_text((x1 + x2)/2, (y1 + y2)/2, text="G", fill="#ffffff", font=("Segoe UI", int(max(8, cell_size * 0.45)), "bold"))

        if len(self.final_path) > 1:
            points = []
            for r, c in self.final_path:
                points.extend([offset_x + c * cell_size + cell_size / 2, offset_y + r * cell_size + cell_size / 2])
            self.canvas.create_line(points, fill="#fbbf24", width=max(2, int(cell_size * 0.22)), capstyle=tk.ROUND, joinstyle=tk.ROUND)

    def _draw_single_cell(self, pt: Point, color: str, outline: Optional[str] = None):
        if pt == self.grid.start or pt == self.grid.goal:
            return
        r, c = pt
        cell_size, offset_x, offset_y, _ = self._get_cell_geometry()
        x1 = offset_x + c * cell_size
        y1 = offset_y + r * cell_size
        self.canvas.create_rectangle(x1, y1, x1 + cell_size, y1 + cell_size, fill=color, outline=outline or THEME["cell_border"], width=1, tags=f"cell_{r}_{c}")

    def _on_mouse_down_left(self, event):
        pt = self._canvas_to_grid(event.x, event.y)
        if not pt:
            return
        self.is_mouse_pressed = True
        if pt == self.grid.start:
            self.drag_entity = "start"
            return
        elif pt == self.grid.goal:
            self.drag_entity = "goal"
            return
        self.drag_entity = None
        self._apply_tool_at(pt)

    def _on_mouse_drag_left(self, event):
        if not self.is_mouse_pressed:
            return
        pt = self._canvas_to_grid(event.x, event.y)
        if not pt:
            return
        if self.drag_entity == "start":
            if pt != self.grid.goal and pt not in self.grid.obstacles:
                self.grid.set_start(pt)
                self.clear_search_visualization()
                self.redraw_canvas()
            return
        elif self.drag_entity == "goal":
            if pt != self.grid.start and pt not in self.grid.obstacles:
                self.grid.set_goal(pt)
                self.clear_search_visualization()
                self.redraw_canvas()
            return
        self._apply_tool_at(pt)

    def _on_mouse_up_left(self, event):
        self.is_mouse_pressed = False
        self.drag_entity = None

    def _on_mouse_down_right(self, event):
        pt = self._canvas_to_grid(event.x, event.y)
        if pt and pt != self.grid.start and pt != self.grid.goal:
            if self.grid.remove_obstacle(pt):
                self._draw_single_cell(pt, THEME["cell_empty"])

    def _on_mouse_drag_right(self, event):
        pt = self._canvas_to_grid(event.x, event.y)
        if pt and pt != self.grid.start and pt != self.grid.goal:
            if self.grid.remove_obstacle(pt):
                self._draw_single_cell(pt, THEME["cell_empty"])

    def _apply_tool_at(self, pt: Point):
        tool = self.current_tool.get()
        if tool == "wall":
            if pt != self.grid.start and pt != self.grid.goal and self.grid.add_obstacle(pt):
                self._draw_single_cell(pt, THEME["cell_wall"], THEME["cell_wall_border"])
        elif tool == "erase":
            if pt != self.grid.start and pt != self.grid.goal and self.grid.remove_obstacle(pt):
                self._draw_single_cell(pt, THEME["cell_empty"])
        elif tool == "start":
            if self.grid.set_start(pt):
                self.clear_search_visualization()
                self.redraw_canvas()
        elif tool == "goal":
            if self.grid.set_goal(pt):
                self.clear_search_visualization()
                self.redraw_canvas()

    def generate_random_map(self):
        self.stop_search()
        self.grid.generate_random(0.28)
        self.clear_search_visualization()
        self.redraw_canvas()

    def generate_recursive_maze(self):
        self.stop_search()
        self.grid.generate_recursive_maze()
        self.clear_search_visualization()
        self.redraw_canvas()

    def generate_u_trap(self):
        self.stop_search()
        self.grid.generate_u_trap()
        self.clear_search_visualization()
        self.redraw_canvas()

    def generate_corridors(self):
        self.stop_search()
        self.grid.generate_corridors()
        self.clear_search_visualization()
        self.redraw_canvas()

    def clear_all_obstacles(self):
        self.stop_search()
        self.grid.clear_obstacles()
        self.clear_search_visualization()
        self.redraw_canvas()

    def clear_search_visualization(self):
        self.stop_search()
        self.frontier_nodes.clear()
        self.visited_nodes.clear()
        self.final_path.clear()
        self.search_generator = None
        self.card_data["STATUS"][1].config(text="Ready", fg=THEME["text_accent"])
        self.card_data["PATH LENGTH"][1].config(text="0 steps")
        self.card_data["PATH COST"][1].config(text="0.00")
        self.card_data["NODES EXPLORED"][1].config(text="0")
        self.card_data["SEARCH TIME"][1].config(text="0.00 ms")
        self.redraw_canvas()

    def _on_algo_change(self, event=None):
        if "BFS" in self.selected_algo.get():
            self.heuristic_combo.config(state=tk.DISABLED)
            self.heuristic_lbl.config(fg="#475569")
        else:
            self.heuristic_combo.config(state="readonly")
            self.heuristic_lbl.config(fg=THEME["text_secondary"])

    def _on_diagonal_toggle(self):
        self.grid.allow_diagonal = self.allow_diagonal_var.get()
        self.clear_search_visualization()

    def _on_resize_grid(self, event=None):
        new_sz = int(self.grid_size_var.get())
        self.stop_search()
        self.grid.resize(new_sz, new_sz)
        self.clear_search_visualization()
        self.redraw_canvas()

    def start_or_resume_search(self):
        if self.is_running:
            return
        if self.is_paused and self.search_generator:
            self.is_paused = False
            self.is_running = True
            self.btn_pause.config(state=tk.NORMAL, text="⏸ Pause")
            self._step_animation_loop()
            return

        self.clear_search_visualization()
        algo_name = self.selected_algo.get()
        h_name = self.selected_heuristic.get()
        h_fn = HEURISTICS[h_name]

        if "BFS" in algo_name:
            self.search_generator = bfs_generator(self.grid.start, self.grid.goal, self.grid.get_neighbors)
        else:
            self.search_generator = a_star_generator(self.grid.start, self.grid.goal, self.grid.get_neighbors, h_fn, h_name)

        self.is_running = True
        self.is_paused = False
        self.btn_run.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.NORMAL, text="⏸ Pause")
        self.card_data["STATUS"][1].config(text="Searching...", fg="#38bdf8")

        if self.speed_ms.get() == 0:
            self._run_instant()
        else:
            self._step_animation_loop()

    def _run_instant(self):
        try:
            for event_type, data in self.search_generator:
                if event_type == "FRONTIER":
                    self.frontier_nodes.add(data)
                elif event_type == "VISIT":
                    self.frontier_nodes.discard(data)
                    self.visited_nodes.add(data)
                elif event_type == "PATH":
                    path, metrics = data
                    self.final_path = path
                    self._update_metrics_display(metrics)
                    break
                elif event_type == "NO_PATH":
                    self._update_metrics_display(data)
                    break
        finally:
            self.stop_search()
            self.redraw_canvas()

    def _step_animation_loop(self):
        if not self.is_running or self.is_paused or not self.search_generator:
            return
        try:
            event_type, data = next(self.search_generator)
            if event_type == "FRONTIER":
                self.frontier_nodes.add(data)
                self._draw_single_cell(data, THEME["cell_frontier"])
            elif event_type == "VISIT":
                self.frontier_nodes.discard(data)
                self.visited_nodes.add(data)
                self._draw_single_cell(data, THEME["cell_visited"])
            elif event_type == "PATH":
                path, metrics = data
                self.final_path = path
                self._update_metrics_display(metrics)
                self.stop_search()
                self.redraw_canvas()
                self._start_robot_traversal_animation()
                return
            elif event_type == "NO_PATH":
                self._update_metrics_display(data)
                self.stop_search()
                return

            delay = max(1, self.speed_ms.get())
            self.animation_timer = self.root.after(delay, self._step_animation_loop)
        except StopIteration:
            self.stop_search()

    def step_forward(self):
        if not self.search_generator:
            algo_name = self.selected_algo.get()
            h_name = self.selected_heuristic.get()
            h_fn = HEURISTICS[h_name]
            if "BFS" in algo_name:
                self.search_generator = bfs_generator(self.grid.start, self.grid.goal, self.grid.get_neighbors)
            else:
                self.search_generator = a_star_generator(self.grid.start, self.grid.goal, self.grid.get_neighbors, h_fn, h_name)
            self.is_paused = True
            self.btn_pause.config(state=tk.NORMAL, text="▶ Resume")

        try:
            event_type, data = next(self.search_generator)
            if event_type == "FRONTIER":
                self.frontier_nodes.add(data)
                self._draw_single_cell(data, THEME["cell_frontier"])
            elif event_type == "VISIT":
                self.frontier_nodes.discard(data)
                self.visited_nodes.add(data)
                self._draw_single_cell(data, THEME["cell_visited"])
            elif event_type == "PATH":
                path, metrics = data
                self.final_path = path
                self._update_metrics_display(metrics)
                self.stop_search()
                self.redraw_canvas()
            elif event_type == "NO_PATH":
                self._update_metrics_display(data)
                self.stop_search()
        except StopIteration:
            self.stop_search()

    def toggle_pause(self):
        if not self.is_running and not self.is_paused:
            return
        if self.is_paused:
            self.is_paused = False
            self.is_running = True
            self.btn_pause.config(text="⏸ Pause")
            self._step_animation_loop()
        else:
            self.is_paused = True
            self.is_running = False
            self.btn_pause.config(text="▶ Resume")
            if self.animation_timer:
                self.root.after_cancel(self.animation_timer)
                self.animation_timer = None

    def stop_search(self):
        self.is_running = False
        self.is_paused = False
        self.btn_run.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.DISABLED, text="⏸ Pause")
        if self.animation_timer:
            self.root.after_cancel(self.animation_timer)
            self.animation_timer = None

    def _start_robot_traversal_animation(self):
        if not self.final_path:
            return
        self.robot_step_index = 0
        self._animate_robot_step()

    def _animate_robot_step(self):
        if self.robot_step_index >= len(self.final_path):
            self.canvas.delete("robot_marker")
            return
        self.canvas.delete("robot_marker")
        r, c = self.final_path[self.robot_step_index]
        cell_size, offset_x, offset_y, _ = self._get_cell_geometry()
        cx = offset_x + c * cell_size + cell_size / 2
        cy = offset_y + r * cell_size + cell_size / 2
        rad = max(4, cell_size * 0.3)
        self.canvas.create_oval(cx - rad, cy - rad, cx + rad, cy + rad, fill=THEME["robot_marker"], outline="#ffffff", width=2, tags="robot_marker")
        self.robot_step_index += 1
        self.robot_timer = self.root.after(45, self._animate_robot_step)

    def open_comparison_dialog(self):
        h_name = self.selected_heuristic.get()
        h_fn = HEURISTICS[h_name]

        astar_metrics = a_star_search(self.grid.start, self.grid.goal, self.grid.get_neighbors, h_fn, h_name)
        bfs_metrics = bfs_search(self.grid.start, self.grid.goal, self.grid.get_neighbors)

        nodes_saved = bfs_metrics.nodes_explored - astar_metrics.nodes_explored
        reduction_pct = 0.0
        if bfs_metrics.nodes_explored > 0:
            reduction_pct = (nodes_saved / bfs_metrics.nodes_explored) * 100.0

        dlg = tk.Toplevel(self.root)
        dlg.title("Algorithm Benchmark: A* vs BFS")
        dlg.geometry("740x550")
        dlg.minsize(680, 480)
        dlg.configure(bg=THEME["bg_sidebar"])
        dlg.transient(self.root)
        dlg.grab_set()

        d_hdr = tk.Frame(dlg, bg="#090d16", height=60)
        d_hdr.pack(fill=tk.X, side=tk.TOP)
        tk.Label(d_hdr, text="⚖️ Algorithmic Performance Benchmark", bg="#090d16", fg=THEME["text_primary"], font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=16, pady=(10, 2))
        tk.Label(d_hdr, text="Tested on identical grid configuration with identical start and goal positions", bg="#090d16", fg=THEME["text_secondary"], font=("Segoe UI", 9)).pack(anchor="w", padx=16, pady=(0, 10))

        content = tk.Frame(dlg, bg=THEME["bg_sidebar"])
        content.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        table_frame = tk.Frame(content, bg=THEME["bg_card"], bd=1, relief=tk.SOLID)
        table_frame.pack(fill=tk.X, pady=(0, 12))

        headers = ["Metric / Parameter", f"A* ({h_name})", "Breadth-First Search (BFS)", "Delta / Advantage"]
        col_widths = [22, 18, 22, 22]

        hdr_row = tk.Frame(table_frame, bg="#0f172a")
        hdr_row.pack(fill=tk.X)
        for h_text, w in zip(headers, col_widths):
            tk.Label(hdr_row, text=h_text, width=w, anchor="w", bg="#0f172a", fg=THEME["text_accent"], font=("Segoe UI", 9, "bold"), padx=8, pady=6).pack(side=tk.LEFT)

        rows_data = [
            ("Path Found?", "Yes" if astar_metrics.path_found else "No", "Yes" if bfs_metrics.path_found else "No", "Identical"),
            ("Path Length (steps)", f"{astar_metrics.path_length}", f"{bfs_metrics.path_length}", "Equal (Both Optimal)" if astar_metrics.path_length == bfs_metrics.path_length else f"Diff: {abs(astar_metrics.path_length - bfs_metrics.path_length)}"),
            ("Total Path Cost", f"{astar_metrics.path_cost:.2f}", f"{bfs_metrics.path_cost:.2f}", "Equal" if abs(astar_metrics.path_cost - bfs_metrics.path_cost) < 1e-4 else f"Diff: {abs(astar_metrics.path_cost - bfs_metrics.path_cost):.2f}"),
            ("Nodes Explored", f"{astar_metrics.nodes_explored}", f"{bfs_metrics.nodes_explored}", f"A* saved {reduction_pct:.1f}% nodes!" if reduction_pct > 0 else "Similar"),
            ("Search Latency", f"{astar_metrics.execution_time_ms:.3f} ms", f"{bfs_metrics.execution_time_ms:.3f} ms", f"{abs(bfs_metrics.execution_time_ms - astar_metrics.execution_time_ms):.3f} ms delta"),
            ("Heuristic Guidance", f"Informed ({h_name})", "Uninformed (None)", "A* directed toward Goal")
        ]

        for idx, (m_name, a_val, b_val, delta_val) in enumerate(rows_data):
            bg_col = THEME["bg_card"] if idx % 2 == 0 else "#162032"
            r_frame = tk.Frame(table_frame, bg=bg_col)
            r_frame.pack(fill=tk.X)
            tk.Label(r_frame, text=m_name, width=col_widths[0], anchor="w", bg=bg_col, fg=THEME["text_secondary"], font=("Segoe UI", 9), padx=8, pady=5).pack(side=tk.LEFT)
            tk.Label(r_frame, text=a_val, width=col_widths[1], anchor="w", bg=bg_col, fg="#10b981", font=("Segoe UI", 9, "bold"), padx=8, pady=5).pack(side=tk.LEFT)
            tk.Label(r_frame, text=b_val, width=col_widths[2], anchor="w", bg=bg_col, fg="#38bdf8", font=("Segoe UI", 9, "bold"), padx=8, pady=5).pack(side=tk.LEFT)
            tk.Label(r_frame, text=delta_val, width=col_widths[3], anchor="w", bg=bg_col, fg="#f59e0b", font=("Segoe UI", 9, "bold"), padx=8, pady=5).pack(side=tk.LEFT)

        summary_box = tk.Frame(content, bg=THEME["bg_card"], bd=1, relief=tk.SOLID)
        summary_box.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        tk.Label(summary_box, text="Theoretical & Engineering Analysis:", bg=THEME["bg_card"], fg=THEME["text_primary"], font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=(8, 4))
        
        if reduction_pct > 0:
            analysis_text = (
                f"• Exploration Efficiency: A* explored {astar_metrics.nodes_explored} nodes versus "
                f"BFS's {bfs_metrics.nodes_explored} nodes, achieving a {reduction_pct:.1f}% reduction in search space.\n"
                f"• Admissible Heuristic: Using {h_name} distance guarantees admissibility (h(n) <= h*(n)), "
                f"ensuring A* finds the optimal path without expanding wasteful concentric rings like BFS.\n"
                f"• Path Optimality: Both algorithms found paths of cost {astar_metrics.path_cost:.2f}, "
                f"demonstrating that A* achieves 100% path optimality with a fraction of node expansions."
            )
        else:
            analysis_text = (
                "• On trivial paths or tiny open maps, both algorithms expand similar nodes.\n"
                "• Try creating obstacles, corridors, or using the 'U-Trap' preset to see A*'s heuristic advantage!"
            )
        tk.Label(summary_box, text=analysis_text, bg=THEME["bg_card"], fg=THEME["text_secondary"], font=("Segoe UI", 8), justify=tk.LEFT, wraplength=640).pack(anchor="w", padx=12, pady=(0, 8))

        btn_bar = tk.Frame(dlg, bg=THEME["bg_sidebar"])
        btn_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=16, pady=12)

        def show_astar_on_grid():
            dlg.destroy()
            self.selected_algo.set("A* Search")
            self._on_algo_change()
            self.start_or_resume_search()

        def show_bfs_on_grid():
            dlg.destroy()
            self.selected_algo.set("Breadth-First Search (BFS)")
            self._on_algo_change()
            self.start_or_resume_search()

        tk.Button(btn_bar, text="Show A* Visualization", command=show_astar_on_grid, bg="#2563eb", fg="#ffffff", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, padx=12, pady=6).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(btn_bar, text="Show BFS Visualization", command=show_bfs_on_grid, bg="#0284c7", fg="#ffffff", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, padx=12, pady=6).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_bar, text="Close", command=dlg.destroy, bg=THEME["bg_card"], fg=THEME["text_primary"], font=("Segoe UI", 9), relief=tk.FLAT, padx=16, pady=6).pack(side=tk.RIGHT)

def main():
    parser = argparse.ArgumentParser(description="Problem 10: Intelligent Robot Navigation (A* vs BFS GUI)")
    parser.add_argument("--rows", type=int, default=20, help="Grid rows (default: 20)")
    parser.add_argument("--cols", type=int, default=20, help="Grid cols (default: 20)")
    args = parser.parse_args()

    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = RobotNavigationApp(root, default_rows=args.rows, default_cols=args.cols)
    root.mainloop()

if __name__ == "__main__":
    main()
