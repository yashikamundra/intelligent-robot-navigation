"""
algorithms.py - Core Search Algorithms for Intelligent Robot Navigation
=======================================================================
Implements:
  - A* Search (with Manhattan, Euclidean, and Octile/Diagonal Heuristics)
  - Breadth-First Search (BFS)
  - Generator-based step-by-step exploration for real-time visualization
  - Pure batch execution for precise microsecond-benchmark metrics
"""

from __future__ import annotations
import math
import time
from collections import deque
import heapq
from typing import Generator, List, Tuple, Dict, Set, Optional, Callable

Point = Tuple[int, int]  # (row, col)

# ---------------------------------------------------------------------------
# Heuristic Functions (All Admissible & Consistent for Grid Topologies)
# ---------------------------------------------------------------------------

def heuristic_manhattan(p1: Point, p2: Point) -> float:
    """
    Manhattan distance: |r1 - r2| + |c1 - c2|
    Admissible and optimal for standard 4-directional (orthogonal) grid movement.
    """
    return float(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))


def heuristic_euclidean(p1: Point, p2: Point) -> float:
    """
    Euclidean distance: sqrt((r1 - r2)^2 + (c1 - c2)^2)
    Straight-line physical distance. Admissible for both 4 and 8-directional movement.
    """
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def heuristic_octile(p1: Point, p2: Point) -> float:
    """
    Octile / Diagonal distance:
    D * (max(dx, dy) - min(dx, dy)) + D2 * min(dx, dy)
    where D = 1.0 (orthogonal) and D2 = sqrt(2) (diagonal).
    Admissible and tightest heuristic for 8-directional movement.
    """
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    d1 = 1.0
    d2 = math.sqrt(2.0)
    return d1 * abs(dx - dy) + d2 * min(dx, dy)


HEURISTICS: Dict[str, Callable[[Point, Point], float]] = {
    "Manhattan": heuristic_manhattan,
    "Euclidean": heuristic_euclidean,
    "Octile / Diagonal": heuristic_octile,
}

# ---------------------------------------------------------------------------
# Search Result Data Class
# ---------------------------------------------------------------------------

class SearchMetrics:
    """Container for search performance metrics."""
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


def reconstruct_path(
    came_from: Dict[Point, Optional[Point]], 
    current: Point
) -> List[Point]:
    """Reconstructs the optimal trajectory from start to goal via parent pointers."""
    path = [current]
    while current in came_from and came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def compute_path_cost(path: List[Point]) -> float:
    """Computes exact cost of path (orthogonal = 1.0, diagonal = sqrt(2))."""
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

# ---------------------------------------------------------------------------
# Search Event Types for Visualization Generator
# ---------------------------------------------------------------------------
# Events yielded:
#   ("FRONTIER", node, parent)
#   ("VISIT", node)
#   ("PATH", path, metrics)
#   ("NO_PATH", metrics)

SearchEvent = Tuple[str, any]

# ---------------------------------------------------------------------------
# A* Search Implementation
# ---------------------------------------------------------------------------

def a_star_search(
    start: Point,
    goal: Point,
    get_neighbors_fn: Callable[[Point], List[Tuple[Point, float]]],
    heuristic_fn: Callable[[Point, Point], float],
    heuristic_name: str = "Manhattan"
) -> SearchMetrics:
    """
    Pure batch A* algorithm execution for high-precision latency benchmarking.
    """
    t0 = time.perf_counter()

    if start == goal:
        dt = (time.perf_counter() - t0) * 1000.0
        return SearchMetrics(
            algorithm="A* Search",
            path_found=True,
            path=[start],
            path_length=0,
            path_cost=0.0,
            nodes_explored=1,
            nodes_in_frontier=0,
            execution_time_ms=dt,
            heuristic_name=heuristic_name
        )

    # Priority queue stores tuples: (f_score, tie_breaker_counter, current_node)
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
                # Tie-breaking nudge: slightly favor nodes closer to goal
                f_val = tentative_g + h_val * 1.0001

                counter += 1
                heapq.heappush(open_heap, (f_val, counter, neighbor))
                open_set_tracker.add(neighbor)

    dt = (time.perf_counter() - t0) * 1000.0
    return SearchMetrics(
        algorithm="A* Search",
        path_found=False,
        path=[],
        path_length=0,
        path_cost=0.0,
        nodes_explored=nodes_explored_count,
        nodes_in_frontier=0,
        execution_time_ms=dt,
        heuristic_name=heuristic_name
    )


def a_star_generator(
    start: Point,
    goal: Point,
    get_neighbors_fn: Callable[[Point], List[Tuple[Point, float]]],
    heuristic_fn: Callable[[Point, Point], float],
    heuristic_name: str = "Manhattan"
) -> Generator[Tuple[str, any], None, None]:
    """
    Generator version of A* Search for real-time visual step animation.
    Yields events:
      - ('FRONTIER', neighbor)
      - ('VISIT', current)
      - ('PATH', (path, metrics))
      - ('NO_PATH', metrics)
    """
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
    m = SearchMetrics(
        algorithm="A* Search",
        path_found=False,
        path=[],
        path_length=0,
        path_cost=0.0,
        nodes_explored=nodes_explored_count,
        nodes_in_frontier=0,
        execution_time_ms=dt,
        heuristic_name=heuristic_name
    )
    yield ("NO_PATH", m)

# ---------------------------------------------------------------------------
# Breadth-First Search (BFS) Implementation
# ---------------------------------------------------------------------------

def bfs_search(
    start: Point,
    goal: Point,
    get_neighbors_fn: Callable[[Point], List[Tuple[Point, float]]]
) -> SearchMetrics:
    """
    Pure batch BFS algorithm execution for high-precision latency benchmarking.
    """
    t0 = time.perf_counter()

    if start == goal:
        dt = (time.perf_counter() - t0) * 1000.0
        return SearchMetrics(
            algorithm="Breadth-First Search (BFS)",
            path_found=True,
            path=[start],
            path_length=0,
            path_cost=0.0,
            nodes_explored=1,
            nodes_in_frontier=0,
            execution_time_ms=dt,
            heuristic_name="None (Uninformed)"
        )

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
    return SearchMetrics(
        algorithm="Breadth-First Search (BFS)",
        path_found=False,
        path=[],
        path_length=0,
        path_cost=0.0,
        nodes_explored=nodes_explored_count,
        nodes_in_frontier=0,
        execution_time_ms=dt,
        heuristic_name="None (Uninformed)"
    )


def bfs_generator(
    start: Point,
    goal: Point,
    get_neighbors_fn: Callable[[Point], List[Tuple[Point, float]]]
) -> Generator[Tuple[str, any], None, None]:
    """
    Generator version of Breadth-First Search for real-time visual step animation.
    Yields events:
      - ('FRONTIER', neighbor)
      - ('VISIT', current)
      - ('PATH', (path, metrics))
      - ('NO_PATH', metrics)
    """
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
    m = SearchMetrics(
        algorithm="Breadth-First Search (BFS)",
        path_found=False,
        path=[],
        path_length=0,
        path_cost=0.0,
        nodes_explored=nodes_explored_count,
        nodes_in_frontier=0,
        execution_time_ms=dt,
        heuristic_name="None (Uninformed)"
    )
    yield ("NO_PATH", m)
