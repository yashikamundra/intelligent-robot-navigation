#!/usr/bin/env python3
"""
robot_navigation.py - Main Entry Point for Intelligent Robot Navigation
========================================================================
Course / Project: Problem 10: Intelligent Robot Navigation
Authorship: Advanced AI Algorithm Engineer & Systems Developer

Description:
  A complete, interactive Python GUI application implementing heuristic A* Search
  and Breadth-First Search (BFS) for intelligent robotic path planning on a 2D occupancy grid.

Features:
  - Interactive grid: Click & drag to draw obstacles, erase, set Start & Goal
  - Admissible heuristics: Manhattan, Euclidean, Octile/Diagonal distance
  - Step-by-step & real-time animated search exploration with speed slider
  - Comprehensive performance dashboard: Path length, path cost, nodes explored, search time (ms)
  - Side-by-side comparison feature benchmarking A* against BFS on identical maps
  - Procedural map generators: Random, Recursive Maze, U-Trap, Alternating Corridors

Usage:
  python robot_navigation.py
  python robot_navigation.py --rows 25 --cols 25
"""

import sys
import argparse
import tkinter as tk

from gui import RobotNavigationApp


def main():
    parser = argparse.ArgumentParser(
        description="Problem 10: Intelligent Robot Navigation (A* vs BFS GUI)"
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=20,
        help="Number of rows in the grid (default: 20, min: 5, max: 50)"
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=20,
        help="Number of columns in the grid (default: 20, min: 5, max: 50)"
    )
    args = parser.parse_args()

    # Validate dimensions
    rows = max(5, min(50, args.rows))
    cols = max(5, min(50, args.cols))

    # Initialize Tkinter root window
    root = tk.Tk()
    
    # Optional Windows DPI awareness enhancement
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    # Launch application
    app = RobotNavigationApp(root, default_rows=rows, default_cols=cols)
    root.mainloop()


if __name__ == "__main__":
    main()
