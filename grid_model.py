"""
grid_model.py - Grid Representation & Map Generators
===================================================
Manages grid state, obstacle layouts, coordinate validation,
and procedural map generation (Random, Recursive Maze, U-Trap, Corridors).
"""

from __future__ import annotations
import math
import random
from typing import Set, Tuple, List, Optional

Point = Tuple[int, int]  # (row, col)

class GridModel:
    """
    Encapsulates 2D occupancy grid state, movement topology, and map generators.
    """
    def __init__(self, rows: int = 20, cols: int = 20):
        self.rows = max(5, rows)
        self.cols = max(5, cols)
        
        # Default start and goal locations
        self.start: Point = (1, 1)
        self.goal: Point = (self.rows - 2, self.cols - 2)
        
        # Set of obstacle coordinates (row, col)
        self.obstacles: Set[Point] = set()
        
        # Movement configuration
        self.allow_diagonal: bool = False

    def in_bounds(self, p: Point) -> bool:
        """Check if a coordinate lies strictly inside grid bounds."""
        r, c = p
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_obstacle(self, p: Point) -> bool:
        """Check if cell is an obstacle."""
        return p in self.obstacles

    def is_passable(self, p: Point) -> bool:
        """Check if cell is within bounds and not an obstacle."""
        return self.in_bounds(p) and (p not in self.obstacles)

    def set_start(self, p: Point) -> bool:
        """Set the start location if valid and distinct from goal."""
        if not self.in_bounds(p) or p == self.goal:
            return False
        self.start = p
        self.obstacles.discard(p)
        return True

    def set_goal(self, p: Point) -> bool:
        """Set the goal location if valid and distinct from start."""
        if not self.in_bounds(p) or p == self.start:
            return False
        self.goal = p
        self.obstacles.discard(p)
        return True

    def add_obstacle(self, p: Point) -> bool:
        """Add obstacle if not start or goal."""
        if self.in_bounds(p) and p != self.start and p != self.goal:
            self.obstacles.add(p)
            return True
        return False

    def remove_obstacle(self, p: Point) -> bool:
        """Remove obstacle if present."""
        if p in self.obstacles:
            self.obstacles.remove(p)
            return True
        return False

    def toggle_obstacle(self, p: Point) -> bool:
        """Toggle obstacle on or off."""
        if p == self.start or p == self.goal or not self.in_bounds(p):
            return False
        if p in self.obstacles:
            self.obstacles.remove(p)
        else:
            self.obstacles.add(p)
        return True

    def clear_obstacles(self) -> None:
        """Clear all obstacles."""
        self.obstacles.clear()

    def resize(self, new_rows: int, new_cols: int) -> None:
        """Resize the grid and clamp start/goal inside bounds."""
        self.rows = max(5, new_rows)
        self.cols = max(5, new_cols)
        
        # Prune out-of-bound obstacles
        self.obstacles = {p for p in self.obstacles if self.in_bounds(p)}
        
        # Clamp start and goal
        sr = min(self.start[0], self.rows - 1)
        sc = min(self.start[1], self.cols - 1)
        self.start = (sr, sc)

        gr = min(self.goal[0], self.rows - 1)
        gc = min(self.goal[1], self.cols - 1)
        if (gr, gc) == self.start:
            gr = max(0, self.rows - 2)
            gc = max(0, self.cols - 2)
        self.goal = (gr, gc)

        self.obstacles.discard(self.start)
        self.obstacles.discard(self.goal)

    def get_neighbors(self, p: Point) -> List[Tuple[Point, float]]:
        """
        Returns list of (neighbor_point, movement_cost) for node p.
        Supports:
          - 4-directional (N, S, E, W with cost 1.0)
          - 8-directional (Orthogonal + Diagonals with cost sqrt(2)),
            with strict corner-cutting prevention (cannot slice through walls).
        """
        r, c = p
        neighbors: List[Tuple[Point, float]] = []

        # 4 Orthogonal directions: Up, Down, Left, Right
        orthogonal = [
            ((-1, 0), 1.0),
            ((1, 0), 1.0),
            ((0, -1), 1.0),
            ((0, 1), 1.0),
        ]

        for (dr, dc), cost in orthogonal:
            nr, nc = r + dr, c + dc
            cand = (nr, nc)
            if self.is_passable(cand):
                neighbors.append((cand, cost))

        # Diagonals (if enabled)
        if self.allow_diagonal:
            diagonals = [
                ((-1, -1), math.sqrt(2.0)),  # NW
                ((-1, 1), math.sqrt(2.0)),   # NE
                ((1, -1), math.sqrt(2.0)),   # SW
                ((1, 1), math.sqrt(2.0)),    # SE
            ]
            for (dr, dc), cost in diagonals:
                nr, nc = r + dr, c + dc
                cand = (nr, nc)
                if not self.is_passable(cand):
                    continue

                # Anti-corner-cutting rule:
                # Do not cut between two diagonally adjacent obstacles
                # Check both orthogonal neighbors that share the corner
                ortho1 = (r + dr, c)
                ortho2 = (r, c + dc)
                if not self.is_passable(ortho1) and not self.is_passable(ortho2):
                    continue

                neighbors.append((cand, cost))

        return neighbors

    # -----------------------------------------------------------------------
    # Procedural Map Generators
    # -----------------------------------------------------------------------

    def generate_random(self, density: float = 0.28) -> None:
        """Fills grid with random obstacles at given density."""
        self.clear_obstacles()
        for r in range(self.rows):
            for c in range(self.cols):
                pt = (r, c)
                if pt != self.start and pt != self.goal:
                    if random.random() < density:
                        self.obstacles.add(pt)

    def generate_u_trap(self) -> None:
        """
        Creates a classic U-shaped convex trap (local minima).
        Forces greedy heuristics and uninformed search to exhibit dramatic differences.
        """
        self.clear_obstacles()
        mid_r = self.rows // 2
        mid_c = self.cols // 2

        # Configure start inside the trap, goal outside
        self.start = (mid_r, mid_c)
        self.goal = (mid_r, self.cols - 3)

        # U-trap barrier opening to the left
        arm_len = max(3, self.rows // 4)
        wall_c = mid_c + 2

        # Vertical back wall
        for r in range(mid_r - arm_len, mid_r + arm_len + 1):
            if 0 <= r < self.rows:
                self.obstacles.add((r, wall_c))

        # Top and bottom horizontal arms
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
        """Generates alternating horizontal slalom corridors."""
        self.clear_obstacles()
        spacing = 3
        for r in range(spacing, self.rows - 1, spacing):
            # Alternating gaps
            if (r // spacing) % 2 == 1:
                # Gap on the right
                for c in range(0, self.cols - 3):
                    self.obstacles.add((r, c))
            else:
                # Gap on the left
                for c in range(3, self.cols):
                    self.obstacles.add((r, c))

        self.start = (1, 1)
        self.goal = (self.rows - 2, self.cols - 2)
        self.obstacles.discard(self.start)
        self.obstacles.discard(self.goal)

    def generate_recursive_maze(self) -> None:
        """
        Generates an authentic maze using Recursive Division.
        Guarantees clear navigable passages and natural dead-ends.
        """
        self.clear_obstacles()
        
        # Add perimeter walls
        for r in range(self.rows):
            self.obstacles.add((r, 0))
            self.obstacles.add((r, self.cols - 1))
        for c in range(self.cols):
            self.obstacles.add((0, c))
            self.obstacles.add((self.rows - 1, c))

        def divide(r1: int, r2: int, c1: int, c2: int) -> None:
            width = c2 - c1
            height = r2 - r1

            if width <= 2 or height <= 2:
                return

            horizontal = (height > width) if height != width else (random.random() < 0.5)

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
