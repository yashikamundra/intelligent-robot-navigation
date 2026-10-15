"""
gui.py - Modern Tkinter User Interface for Robot Navigation
===========================================================
Provides:
  - Interactive drag-to-paint grid canvas with dynamic scaling
  - Animated search exploration (Visited, Frontier, Shortest Path, Robot traversal)
  - Real-time performance dashboard (Path length, cost, nodes explored, search time)
  - Side-by-side comparison modal between A* Search and BFS
  - Map presets (Random, Recursive Maze, U-Trap, Corridors)
"""

from __future__ import annotations
import math
import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Dict, Set, List, Tuple

from algorithms import (
    HEURISTICS,
    SearchMetrics,
    a_star_generator,
    bfs_generator,
    a_star_search,
    bfs_search,
    Point
)
from grid_model import GridModel


# ---------------------------------------------------------------------------
# Color Palette (Modern Dark Theme)
# ---------------------------------------------------------------------------
THEME = {
    "bg_main": "#0f172a",         # Deep slate dark
    "bg_card": "#1e293b",         # Card background
    "bg_card_border": "#334155",  # Border for cards
    "bg_sidebar": "#111827",      # Left sidebar background
    "text_primary": "#f8fafc",    # Pure crisp white
    "text_secondary": "#94a3b8",  # Muted slate
    "text_accent": "#38bdf8",     # Cyan highlight
    
    # Grid cell colors
    "cell_empty": "#1e293b",
    "cell_border": "#334155",
    "cell_wall": "#020617",
    "cell_wall_border": "#475569",
    "cell_start": "#10b981",      # Emerald green
    "cell_goal": "#ef4444",       # Ruby red
    "cell_frontier": "#8b5cf6",   # Electric purple
    "cell_visited": "#3b82f6",    # Azure blue
    "cell_visited_bfs": "#0284c7",# BFS contrast color
    "cell_path": "#fbbf24",       # Golden amber
    "cell_path_glow": "#fef08a",
    "robot_marker": "#f43f5e",    # Rose red robot tracer
}


class RobotNavigationApp:
    """
    Main Tkinter GUI Controller for Problem 10: Intelligent Robot Navigation.
    """
    def __init__(self, root: tk.Tk, default_rows: int = 20, default_cols: int = 20):
        self.root = root
        self.root.title("Problem 10: Intelligent Robot Navigation (A* vs BFS)")
        self.root.geometry("1280x820")
        self.root.minsize(1050, 700)
        self.root.configure(bg=THEME["bg_main"])

        # Grid Model State
        self.grid = GridModel(rows=default_rows, cols=default_cols)
        
        # Interactive Tool State
        self.current_tool = tk.StringVar(value="wall")  # 'start', 'goal', 'wall', 'erase'
        self.is_mouse_pressed = False
        self.drag_entity: Optional[str] = None  # 'start' or 'goal' when dragged directly
        
        # Algorithm Configuration
        self.selected_algo = tk.StringVar(value="A* Search")
        self.selected_heuristic = tk.StringVar(value="Manhattan")
        self.allow_diagonal_var = tk.BooleanVar(value=False)
        self.speed_ms = tk.IntVar(value=15)  # Delay in milliseconds per step
        self.grid_size_var = tk.IntVar(value=20)
        
        # Runtime Search & Animation State
        self.search_generator = None
        self.is_running = False
        self.is_paused = False
        self.animation_timer = None
        self.robot_timer = None
        
        # Visual Node Tracking
        self.frontier_nodes: Set[Point] = set()
        self.visited_nodes: Set[Point] = set()
        self.final_path: List[Point] = []
        self.last_metrics: Optional[SearchMetrics] = None
        self.robot_step_index: int = 0

        # Build UI Components
        self._configure_styles()
        self._build_layout()
        self._bind_events()

        # Initial Draw
        self.root.update_idletasks()
        self.redraw_canvas()

    # -----------------------------------------------------------------------
    # Styling Configuration
    # -----------------------------------------------------------------------
    def _configure_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")

        # Generic Frame & Label Styling
        style.configure("Dark.TFrame", background=THEME["bg_sidebar"])
        style.configure("Card.TFrame", background=THEME["bg_card"])
        style.configure("Main.TFrame", background=THEME["bg_main"])
        
        style.configure(
            "Header.TLabel",
            background=THEME["bg_sidebar"],
            foreground=THEME["text_primary"],
            font=("Segoe UI", 13, "bold")
        )
        style.configure(
            "SubHeader.TLabel",
            background=THEME["bg_sidebar"],
            foreground=THEME["text_secondary"],
            font=("Segoe UI", 9)
        )
        style.configure(
            "CardTitle.TLabel",
            background=THEME["bg_card"],
            foreground=THEME["text_secondary"],
            font=("Segoe UI", 9, "bold")
        )
        style.configure(
            "CardVal.TLabel",
            background=THEME["bg_card"],
            foreground=THEME["text_primary"],
            font=("Segoe UI", 13, "bold")
        )
        style.configure(
            "Dark.TLabel",
            background=THEME["bg_sidebar"],
            foreground=THEME["text_primary"],
            font=("Segoe UI", 10)
        )

        # Combobox Styling
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", THEME["bg_card"])],
            selectbackground=[("readonly", THEME["bg_card"])],
            selectforeground=[("readonly", THEME["text_primary"])],
            background=[("readonly", THEME["bg_card_border"])]
        )

    # -----------------------------------------------------------------------
    # Layout Assembly
    # -----------------------------------------------------------------------
    def _build_layout(self):
        # Top Header Bar
        header_bar = tk.Frame(self.root, bg="#090d16", height=50)
        header_bar.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header_bar,
            text="🤖 Intelligent Robot Navigation System",
            bg="#090d16",
            fg=THEME["text_primary"],
            font=("Segoe UI", 14, "bold")
        )
        title_lbl.pack(side=tk.LEFT, padx=16, pady=8)

        subtitle_lbl = tk.Label(
            header_bar,
            text="Heuristic A* Search vs Breadth-First Search (BFS) Benchmark",
            bg="#090d16",
            fg=THEME["text_accent"],
            font=("Segoe UI", 10)
        )
        subtitle_lbl.pack(side=tk.LEFT, padx=4, pady=8)

        # Main Workspace Container (Sidebar on left, Canvas & Metrics on right)
        workspace = tk.Frame(self.root, bg=THEME["bg_main"])
        workspace.pack(fill=tk.BOTH, expand=True)

        # Left Sidebar (Controls & Settings)
        sidebar_frame = tk.Frame(workspace, bg=THEME["bg_sidebar"], width=320)
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        sidebar_frame.pack_propagate(False)

        # Wrap sidebar in scrollable frame if needed
        self._build_sidebar_controls(sidebar_frame)

        # Center / Right Area
        right_panel = tk.Frame(workspace, bg=THEME["bg_main"])
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Top Metrics Cards Row
        self._build_metrics_panel(right_panel)

        # Grid Canvas Area
        canvas_container = tk.Frame(right_panel, bg=THEME["bg_card"], bd=1, relief=tk.SOLID)
        canvas_container.pack(fill=tk.BOTH, expand=True, pady=(8, 4))

        self.canvas = tk.Canvas(
            canvas_container,
            bg=THEME["bg_main"],
            highlightthickness=0,
            cursor="crosshair"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Legend Bar at Bottom
        self._build_legend_bar(right_panel)

    def _build_sidebar_controls(self, parent: tk.Frame):
        """Constructs buttons, sliders, and options in left control panel."""
        canvas_side = tk.Canvas(parent, bg=THEME["bg_sidebar"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas_side.yview)
        scrollable_frame = tk.Frame(canvas_side, bg=THEME["bg_sidebar"])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_side.configure(scrollregion=canvas_side.bbox("all"))
        )
        canvas_side.create_window((0, 0), window=scrollable_frame, anchor="nw", width=305)
        canvas_side.configure(yscrollcommand=scrollbar.set)

        canvas_side.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        pad_x = 12
        pad_y = 6

        # --- Section 1: Interactive Palette ---
        sec1 = self._create_section(scrollable_frame, "1. INTERACTIVE TOOLS")
        
        tools_row = tk.Frame(sec1, bg=THEME["bg_sidebar"])
        tools_row.pack(fill=tk.X, padx=pad_x, pady=pad_y)

        tools = [
            ("🟢 Start", "start", "#10b981"),
            ("🎯 Goal", "goal", "#ef4444"),
            ("🧱 Wall", "wall", "#64748b"),
            ("🧹 Erase", "erase", "#94a3b8"),
        ]
        for idx, (label, val, col) in enumerate(tools):
            rb = tk.Radiobutton(
                tools_row,
                text=label,
                value=val,
                variable=self.current_tool,
                indicatoron=False,
                bg=THEME["bg_card"],
                fg=THEME["text_primary"],
                selectcolor="#2563eb",
                activebackground="#3b82f6",
                activeforeground="#ffffff",
                font=("Segoe UI", 8, "bold"),
                relief=tk.FLAT,
                bd=2,
                padx=6,
                pady=4
            )
            rb.grid(row=idx // 2, column=idx % 2, padx=3, pady=3, sticky="nsew")
        tools_row.grid_columnconfigure(0, weight=1)
        tools_row.grid_columnconfigure(1, weight=1)

        hint_lbl = tk.Label(
            sec1,
            text="Tip: Left-drag draws, Right-drag erases\nDrag Start / Goal directly to reposition",
            bg=THEME["bg_sidebar"],
            fg=THEME["text_secondary"],
            font=("Segoe UI", 7, "italic"),
            justify=tk.CENTER
        )
        hint_lbl.pack(pady=(0, 4))

        # --- Section 2: Algorithm Selection ---
        sec2 = self._create_section(scrollable_frame, "2. ALGORITHM & HEURISTIC")
        
        tk.Label(
            sec2, text="Search Algorithm:", bg=THEME["bg_sidebar"],
            fg=THEME["text_secondary"], font=("Segoe UI", 8, "bold")
        ).pack(anchor="w", padx=pad_x, pady=(pad_y, 2))

        algo_combo = ttk.Combobox(
            sec2,
            textvariable=self.selected_algo,
            values=["A* Search", "Breadth-First Search (BFS)"],
            state="readonly",
            font=("Segoe UI", 9)
        )
        algo_combo.pack(fill=tk.X, padx=pad_x, pady=(0, 6))
        algo_combo.bind("<<ComboboxSelected>>", self._on_algo_change)

        self.heuristic_lbl = tk.Label(
            sec2, text="A* Heuristic Function:", bg=THEME["bg_sidebar"],
            fg=THEME["text_secondary"], font=("Segoe UI", 8, "bold")
        )
        self.heuristic_lbl.pack(anchor="w", padx=pad_x, pady=(2, 2))

        self.heuristic_combo = ttk.Combobox(
            sec2,
            textvariable=self.selected_heuristic,
            values=list(HEURISTICS.keys()),
            state="readonly",
            font=("Segoe UI", 9)
        )
        self.heuristic_combo.pack(fill=tk.X, padx=pad_x, pady=(0, 6))

        # Movement Topology
        diag_check = tk.Checkbutton(
            sec2,
            text="Allow 8-Directional (Diagonal)",
            variable=self.allow_diagonal_var,
            command=self._on_diagonal_toggle,
            bg=THEME["bg_sidebar"],
            fg=THEME["text_primary"],
            selectcolor=THEME["bg_card"],
            activebackground=THEME["bg_sidebar"],
            activeforeground=THEME["text_accent"],
            font=("Segoe UI", 8)
        )
        diag_check.pack(anchor="w", padx=pad_x, pady=(0, 4))

        # Grid Dimension
        dim_frame = tk.Frame(sec2, bg=THEME["bg_sidebar"])
        dim_frame.pack(fill=tk.X, padx=pad_x, pady=(4, 6))
        tk.Label(
            dim_frame, text="Grid Size:", bg=THEME["bg_sidebar"],
            fg=THEME["text_secondary"], font=("Segoe UI", 8, "bold")
        ).pack(side=tk.LEFT)
        
        dim_combo = ttk.Combobox(
            dim_frame,
            textvariable=self.grid_size_var,
            values=[15, 20, 25, 30],
            width=6,
            state="readonly"
        )
        dim_combo.pack(side=tk.RIGHT)
        dim_combo.bind("<<ComboboxSelected>>", self._on_resize_grid)

        # --- Section 3: Procedural Map Presets ---
        sec3 = self._create_section(scrollable_frame, "3. PROCEDURAL MAP PRESETS")
        presets_frame = tk.Frame(sec3, bg=THEME["bg_sidebar"])
        presets_frame.pack(fill=tk.X, padx=pad_x, pady=pad_y)

        preset_btns = [
            ("🎲 Random Map", self.generate_random_map),
            ("🌀 Recursive Maze", self.generate_recursive_maze),
            ("🧲 U-Trap (Local Minima)", self.generate_u_trap),
            ("〰️ Alternating Corridors", self.generate_corridors),
            ("🗑️ Clear Walls", self.clear_all_obstacles),
            ("🔄 Clear Search", self.clear_search_visualization),
        ]
        for idx, (lbl, cmd) in enumerate(preset_btns):
            b = tk.Button(
                presets_frame,
                text=lbl,
                command=cmd,
                bg=THEME["bg_card"],
                fg=THEME["text_primary"],
                activebackground="#334155",
                activeforeground="#ffffff",
                relief=tk.FLAT,
                font=("Segoe UI", 8),
                padx=4,
                pady=4
            )
            b.grid(row=idx // 2, column=idx % 2, padx=2, pady=2, sticky="nsew")
        presets_frame.grid_columnconfigure(0, weight=1)
        presets_frame.grid_columnconfigure(1, weight=1)

        # --- Section 4: Execution & Animation Controls ---
        sec4 = self._create_section(scrollable_frame, "4. EXECUTION CONTROLS")
        
        # Speed Slider
        speed_box = tk.Frame(sec4, bg=THEME["bg_sidebar"])
        speed_box.pack(fill=tk.X, padx=pad_x, pady=(pad_y, 4))
        tk.Label(
            speed_box, text="Animation Speed:", bg=THEME["bg_sidebar"],
            fg=THEME["text_secondary"], font=("Segoe UI", 8, "bold")
        ).pack(anchor="w")

        speed_scale = tk.Scale(
            speed_box,
            from_=0,
            to=60,
            orient=tk.HORIZONTAL,
            variable=self.speed_ms,
            bg=THEME["bg_sidebar"],
            fg=THEME["text_primary"],
            troughcolor=THEME["bg_card"],
            highlightthickness=0,
            showvalue=True
        )
        speed_scale.pack(fill=tk.X)
        tk.Label(
            speed_box, text="0 ms (Instant) ⟵ ⟶ 60 ms (Slow)",
            bg=THEME["bg_sidebar"], fg=THEME["text_secondary"],
            font=("Segoe UI", 7)
        ).pack(anchor="center")

        # Action Buttons
        self.btn_run = tk.Button(
            sec4,
            text="▶ START SEARCH",
            command=self.start_or_resume_search,
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            pady=6
        )
        self.btn_run.pack(fill=tk.X, padx=pad_x, pady=(8, 3))

        playback_row = tk.Frame(sec4, bg=THEME["bg_sidebar"])
        playback_row.pack(fill=tk.X, padx=pad_x, pady=3)

        self.btn_pause = tk.Button(
            playback_row,
            text="⏸ Pause",
            command=self.toggle_pause,
            bg=THEME["bg_card"],
            fg=THEME["text_primary"],
            activebackground="#334155",
            relief=tk.FLAT,
            font=("Segoe UI", 8, "bold"),
            state=tk.DISABLED
        )
        self.btn_pause.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.btn_step = tk.Button(
            playback_row,
            text="⏭ Step",
            command=self.step_forward,
            bg=THEME["bg_card"],
            fg=THEME["text_primary"],
            activebackground="#334155",
            relief=tk.FLAT,
            font=("Segoe UI", 8, "bold")
        )
        self.btn_step.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        # --- Section 5: Direct Comparison Feature ---
        sec5 = self._create_section(scrollable_frame, "5. A* VS BFS COMPARISON")
        
        self.btn_compare = tk.Button(
            sec5,
            text="⚖️ COMPARE A* VS BFS",
            command=self.open_comparison_dialog,
            bg="#7c3aed",
            fg="#ffffff",
            activebackground="#6d28d9",
            activeforeground="#ffffff",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            pady=6
        )
        self.btn_compare.pack(fill=tk.X, padx=pad_x, pady=(pad_y, pad_y + 4))

    def _create_section(self, parent: tk.Widget, title: str) -> tk.Frame:
        """Helper to create visually structured collapsible section headers."""
        frame = tk.Frame(parent, bg=THEME["bg_sidebar"], bd=0)
        frame.pack(fill=tk.X, pady=(8, 0))

        lbl = tk.Label(
            frame,
            text=title,
            bg=THEME["bg_sidebar"],
            fg=THEME["text_accent"],
            font=("Segoe UI", 8, "bold")
        )
        lbl.pack(anchor="w", padx=12, pady=(4, 2))
        
        sep = tk.Frame(frame, bg="#1e293b", height=1)
        sep.pack(fill=tk.X, padx=12, pady=(0, 4))
        return frame

    # -----------------------------------------------------------------------
    # Metrics Dashboard Cards
    # -----------------------------------------------------------------------
    def _build_metrics_panel(self, parent: tk.Frame):
        """Creates top row of dashboard cards for real-time telemetry."""
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

        for idx, (title, default_val, val_color) in enumerate(cards):
            card = tk.Frame(metrics_bar, bg=THEME["bg_card"], bd=1, relief=tk.SOLID)
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)

            t_lbl = tk.Label(
                card, text=title, bg=THEME["bg_card"],
                fg=THEME["text_secondary"], font=("Segoe UI", 7, "bold")
            )
            t_lbl.pack(anchor="w", padx=6, pady=(4, 0))

            v_lbl = tk.Label(
                card, text=default_val, bg=THEME["bg_card"],
                fg=val_color, font=("Segoe UI", 11, "bold")
            )
            v_lbl.pack(anchor="w", padx=6, pady=(0, 4))

            self.card_data[title] = (t_lbl, v_lbl)

    def _update_metrics_display(self, m: SearchMetrics):
        """Updates dashboard metric cards with latest search telemetry."""
        self.last_metrics = m
        status_text = "Path Found" if m.path_found else "No Path Found"
        status_color = "#10b981" if m.path_found else "#ef4444"

        self.card_data["STATUS"][1].config(text=status_text, fg=status_color)
        self.card_data["PATH LENGTH"][1].config(text=f"{m.path_length} steps")
        self.card_data["PATH COST"][1].config(text=f"{m.path_cost:.2f}")
        self.card_data["NODES EXPLORED"][1].config(
            text=f"{m.nodes_explored} nodes"
        )
        self.card_data["SEARCH TIME"][1].config(text=f"{m.execution_time_ms:.2f} ms")
        
        algo_title = "A*" if "A*" in m.algorithm else "BFS"
        if m.heuristic_name and m.heuristic_name != "None (Uninformed)":
            algo_title += f" ({m.heuristic_name})"
        self.card_data["ACTIVE ALGO"][1].config(text=algo_title)

    # -----------------------------------------------------------------------
    # Legend Bar
    # -----------------------------------------------------------------------
    def _build_legend_bar(self, parent: tk.Frame):
        """Creates the color-coded visual legend below the grid."""
        legend = tk.Frame(parent, bg=THEME["bg_sidebar"], height=28)
        legend.pack(fill=tk.X, side=tk.BOTTOM, pady=(4, 0))

        items = [
            ("Start Node", THEME["cell_start"]),
            ("Goal Node", THEME["cell_goal"]),
            ("Obstacle / Wall", THEME["cell_wall"]),
            ("Explored (Closed)", THEME["cell_visited"]),
            ("Frontier (Open)", THEME["cell_frontier"]),
            ("Final Optimal Path", THEME["cell_path"]),
        ]

        for label, color in items:
            frame = tk.Frame(legend, bg=THEME["bg_sidebar"])
            frame.pack(side=tk.LEFT, padx=10, pady=4)

            swatch = tk.Canvas(frame, width=12, height=12, bg=color, highlightthickness=1, highlightbackground="#ffffff")
            swatch.pack(side=tk.LEFT, padx=(0, 4))

            lbl = tk.Label(
                frame, text=label, bg=THEME["bg_sidebar"],
                fg=THEME["text_secondary"], font=("Segoe UI", 8)
            )
            lbl.pack(side=tk.LEFT)

    # -----------------------------------------------------------------------
    # Canvas Event Handlers & Dynamic Grid Rendering
    # -----------------------------------------------------------------------
    def _bind_events(self):
        self.canvas.bind("<Configure>", lambda e: self.redraw_canvas())
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down_left)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag_left)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up_left)
        
        # Right click for fast erasing
        self.canvas.bind("<ButtonPress-3>", self._on_mouse_down_right)
        self.canvas.bind("<B3-Motion>", self._on_mouse_drag_right)

        # Global Keyboard Shortcuts
        self.root.bind("<space>", lambda e: self.start_or_resume_search())
        self.root.bind("<Escape>", lambda e: self.clear_search_visualization())
        self.root.bind("c", lambda e: self.open_comparison_dialog())
        self.root.bind("C", lambda e: self.open_comparison_dialog())

    def _get_cell_geometry(self) -> Tuple[float, float, float, float]:
        """Calculates square cell size and offsets to keep grid centered in canvas."""
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            return 20.0, 0.0, 0.0, 20.0

        cell_w = w / self.grid.cols
        cell_h = h / self.grid.rows
        cell_size = min(cell_w, cell_h) * 0.96  # Slight margin
        
        offset_x = (w - (cell_size * self.grid.cols)) / 2.0
        offset_y = (h - (cell_size * self.grid.rows)) / 2.0
        return cell_size, offset_x, offset_y, w

    def _canvas_to_grid(self, x: float, y: float) -> Optional[Point]:
        """Converts canvas pixel coordinates into grid (row, col)."""
        cell_size, offset_x, offset_y, _ = self._get_cell_geometry()
        r = int((y - offset_y) // cell_size)
        c = int((x - offset_x) // cell_size)
        if 0 <= r < self.grid.rows and 0 <= c < self.grid.cols:
            return (r, c)
        return None

    def redraw_canvas(self):
        """Full redraw of the entire grid, obstacles, visited cells, and path."""
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

                # Determine Cell Color & Inset
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
                    x1, y1, x2, y2,
                    fill=color,
                    outline=THEME["cell_border"] if tag != "wall" else THEME["cell_wall_border"],
                    width=1,
                    tags=(tag, f"cell_{r}_{c}")
                )

                # Icons / Letter badges for Start and Goal
                if pt == self.grid.start:
                    self.canvas.create_text(
                        (x1 + x2) / 2, (y1 + y2) / 2,
                        text="S", fill="#ffffff",
                        font=("Segoe UI", int(max(9, cell_size * 0.45)), "bold")
                    )
                elif pt == self.grid.goal:
                    self.canvas.create_text(
                        (x1 + x2) / 2, (y1 + y2) / 2,
                        text="G", fill="#ffffff",
                        font=("Segoe UI", int(max(9, cell_size * 0.45)), "bold")
                    )

        # Draw connecting line through optimal path if present
        if len(self.final_path) > 1:
            points = []
            for r, c in self.final_path:
                cx = offset_x + c * cell_size + cell_size / 2
                cy = offset_y + r * cell_size + cell_size / 2
                points.extend([cx, cy])
            self.canvas.create_line(
                points,
                fill="#fbbf24",
                width=max(2, int(cell_size * 0.22)),
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND,
                tags="path_line"
            )

    def _draw_single_cell(self, pt: Point, color: str, outline: Optional[str] = None):
        """Fast incremental update for a single cell during animation."""
        r, c = pt
        if pt == self.grid.start or pt == self.grid.goal:
            return  # Don't overwrite start and goal badges
        
        cell_size, offset_x, offset_y, _ = self._get_cell_geometry()
        x1 = offset_x + c * cell_size
        y1 = offset_y + r * cell_size
        x2 = x1 + cell_size
        y2 = y1 + cell_size

        self.canvas.create_rectangle(
            x1, y1, x2, y2,
            fill=color,
            outline=outline or THEME["cell_border"],
            width=1,
            tags=f"cell_{r}_{c}"
        )

    # -----------------------------------------------------------------------
    # Mouse Interaction (Drag-to-Paint & Quick Erase)
    # -----------------------------------------------------------------------
    def _on_mouse_down_left(self, event):
        pt = self._canvas_to_grid(event.x, event.y)
        if not pt:
            return
        self.is_mouse_pressed = True

        # Check if user clicked directly on Start or Goal to drag them
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
        """Applies current tool (wall, erase, start, goal) to clicked cell."""
        tool = self.current_tool.get()
        if tool == "wall":
            if pt != self.grid.start and pt != self.grid.goal:
                if self.grid.add_obstacle(pt):
                    self._draw_single_cell(pt, THEME["cell_wall"], THEME["cell_wall_border"])
        elif tool == "erase":
            if pt != self.grid.start and pt != self.grid.goal:
                if self.grid.remove_obstacle(pt):
                    self._draw_single_cell(pt, THEME["cell_empty"])
        elif tool == "start":
            if self.grid.set_start(pt):
                self.clear_search_visualization()
                self.redraw_canvas()
        elif tool == "goal":
            if self.grid.set_goal(pt):
                self.clear_search_visualization()
                self.redraw_canvas()

    # -----------------------------------------------------------------------
    # Map Presets & Operations
    # -----------------------------------------------------------------------
    def generate_random_map(self):
        self.stop_search()
        self.grid.generate_random(density=0.28)
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
        """Clears explored/path marks while keeping obstacles intact."""
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

    # -----------------------------------------------------------------------
    # Search Execution & Step Animation Loop
    # -----------------------------------------------------------------------
    def _on_algo_change(self, event=None):
        algo = self.selected_algo.get()
        if "BFS" in algo:
            self.heuristic_combo.config(state=tk.DISABLED)
            self.heuristic_lbl.config(fg="#475569")
        else:
            self.heuristic_combo.config(state="readonly")
            self.heuristic_lbl.config(fg=THEME["text_secondary"])

    def _on_diagonal_toggle(self):
        self.grid.allow_diagonal = self.allow_diagonal_var.get()
        self.clear_search_visualization()

    def _on_resize_grid(self, event=None):
        new_size = int(self.grid_size_var.get())
        self.stop_search()
        self.grid.resize(new_size, new_size)
        self.clear_search_visualization()
        self.redraw_canvas()

    def start_or_resume_search(self):
        """Starts a fresh search or resumes a paused visualization."""
        if self.is_running:
            return

        if self.is_paused and self.search_generator:
            self.is_paused = False
            self.is_running = True
            self.btn_pause.config(state=tk.NORMAL, text="⏸ Pause")
            self._step_animation_loop()
            return

        # Prepare new search generator
        self.clear_search_visualization()
        algo_name = self.selected_algo.get()
        h_name = self.selected_heuristic.get()
        h_fn = HEURISTICS[h_name]

        if "BFS" in algo_name:
            self.search_generator = bfs_generator(
                self.grid.start,
                self.grid.goal,
                self.grid.get_neighbors
            )
        else:
            self.search_generator = a_star_generator(
                self.grid.start,
                self.grid.goal,
                self.grid.get_neighbors,
                h_fn,
                h_name
            )

        self.is_running = True
        self.is_paused = False
        self.btn_run.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.NORMAL, text="⏸ Pause")
        self.card_data["STATUS"][1].config(text="Searching...", fg="#38bdf8")

        # Instant mode (0 ms delay): run directly to completion
        if self.speed_ms.get() == 0:
            self._run_instant()
        else:
            self._step_animation_loop()

    def _run_instant(self):
        """Runs the search to completion in batch mode without frame delay."""
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
        """Non-blocking animation tick using tkinter after() loop."""
        if not self.is_running or self.is_paused or not self.search_generator:
            return

        try:
            # Yield next search event
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
        """Executes a single step for educational inspection and debugging."""
        if not self.search_generator:
            # Initialize generator
            algo_name = self.selected_algo.get()
            h_name = self.selected_heuristic.get()
            h_fn = HEURISTICS[h_name]

            if "BFS" in algo_name:
                self.search_generator = bfs_generator(
                    self.grid.start,
                    self.grid.goal,
                    self.grid.get_neighbors
                )
            else:
                self.search_generator = a_star_generator(
                    self.grid.start,
                    self.grid.goal,
                    self.grid.get_neighbors,
                    h_fn,
                    h_name
                )
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
        """Animates a robot traversing along the computed shortest path."""
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
        radius = max(4, cell_size * 0.3)

        self.canvas.create_oval(
            cx - radius, cy - radius, cx + radius, cy + radius,
            fill=THEME["robot_marker"],
            outline="#ffffff",
            width=2,
            tags="robot_marker"
        )
        self.robot_step_index += 1
        self.robot_timer = self.root.after(45, self._animate_robot_step)

    # -----------------------------------------------------------------------
    # Side-by-Side Comparison Feature (A* vs BFS)
    # -----------------------------------------------------------------------
    def open_comparison_dialog(self):
        """
        Executes both A* (with current heuristic) and BFS on the current grid,
        and presents a side-by-side comparison modal with metrics and analysis.
        """
        # Execute both algorithms headlessly for pure performance benchmarking
        h_name = self.selected_heuristic.get()
        h_fn = HEURISTICS[h_name]

        # 1. Benchmark A*
        astar_metrics = a_star_search(
            self.grid.start,
            self.grid.goal,
            self.grid.get_neighbors,
            h_fn,
            h_name
        )

        # 2. Benchmark BFS
        bfs_metrics = bfs_search(
            self.grid.start,
            self.grid.goal,
            self.grid.get_neighbors
        )

        # Calculate efficiency deltas
        nodes_saved = bfs_metrics.nodes_explored - astar_metrics.nodes_explored
        reduction_pct = 0.0
        if bfs_metrics.nodes_explored > 0:
            reduction_pct = (nodes_saved / bfs_metrics.nodes_explored) * 100.0

        # Create Top-level Modal Dialog
        dlg = tk.Toplevel(self.root)
        dlg.title("Algorithm Comparison: A* Search vs Breadth-First Search (BFS)")
        dlg.geometry("740x560")
        dlg.minsize(680, 500)
        dlg.configure(bg=THEME["bg_sidebar"])
        dlg.transient(self.root)
        dlg.grab_set()

        # Dialog Header
        d_hdr = tk.Frame(dlg, bg="#090d16", height=60)
        d_hdr.pack(fill=tk.X, side=tk.TOP)

        tk.Label(
            d_hdr,
            text="⚖️ Algorithmic Performance Benchmark",
            bg="#090d16", fg=THEME["text_primary"],
            font=("Segoe UI", 13, "bold")
        ).pack(anchor="w", padx=16, pady=(10, 2))

        tk.Label(
            d_hdr,
            text="Tested on identical grid configuration with identical start and goal positions",
            bg="#090d16", fg=THEME["text_secondary"],
            font=("Segoe UI", 9)
        ).pack(anchor="w", padx=16, pady=(0, 10))

        # Comparison Content Frame
        content = tk.Frame(dlg, bg=THEME["bg_sidebar"])
        content.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        # Comparison Table
        table_frame = tk.Frame(content, bg=THEME["bg_card"], bd=1, relief=tk.SOLID)
        table_frame.pack(fill=tk.X, pady=(0, 12))

        # Table Column Headers
        headers = ["Metric / Parameter", f"A* ({h_name})", "Breadth-First Search (BFS)", "Delta / Advantage"]
        col_widths = [22, 18, 22, 22]

        hdr_row = tk.Frame(table_frame, bg="#0f172a")
        hdr_row.pack(fill=tk.X)
        for h_text, w in zip(headers, col_widths):
            tk.Label(
                hdr_row, text=h_text, width=w, anchor="w",
                bg="#0f172a", fg=THEME["text_accent"],
                font=("Segoe UI", 9, "bold"), padx=8, pady=6
            ).pack(side=tk.LEFT)

        rows_data = [
            ("Path Found?",
             "Yes" if astar_metrics.path_found else "No",
             "Yes" if bfs_metrics.path_found else "No",
             "Identical Solvability"),
            ("Path Length (steps)",
             f"{astar_metrics.path_length}",
             f"{bfs_metrics.path_length}",
             "Equal (Both Optimal)" if astar_metrics.path_length == bfs_metrics.path_length else f"Diff: {abs(astar_metrics.path_length - bfs_metrics.path_length)}"),
            ("Total Path Cost",
             f"{astar_metrics.path_cost:.2f}",
             f"{bfs_metrics.path_cost:.2f}",
             "Equal" if abs(astar_metrics.path_cost - bfs_metrics.path_cost) < 1e-4 else f"Diff: {abs(astar_metrics.path_cost - bfs_metrics.path_cost):.2f}"),
            ("Nodes Explored",
             f"{astar_metrics.nodes_explored}",
             f"{bfs_metrics.nodes_explored}",
             f"A* saved {reduction_pct:.1f}% nodes!" if reduction_pct > 0 else "Similar"),
            ("Search Latency",
             f"{astar_metrics.execution_time_ms:.3f} ms",
             f"{bfs_metrics.execution_time_ms:.3f} ms",
             f"{abs(bfs_metrics.execution_time_ms - astar_metrics.execution_time_ms):.3f} ms delta"),
            ("Heuristic Guidance",
             f"Informed ({h_name})",
             "Uninformed (None)",
             "A* directed toward Goal")
        ]

        for idx, (m_name, a_val, b_val, delta_val) in enumerate(rows_data):
            bg_col = THEME["bg_card"] if idx % 2 == 0 else "#162032"
            r_frame = tk.Frame(table_frame, bg=bg_col)
            r_frame.pack(fill=tk.X)

            tk.Label(r_frame, text=m_name, width=col_widths[0], anchor="w",
                     bg=bg_col, fg=THEME["text_secondary"], font=("Segoe UI", 9), padx=8, pady=5).pack(side=tk.LEFT)
            tk.Label(r_frame, text=a_val, width=col_widths[1], anchor="w",
                     bg=bg_col, fg="#10b981", font=("Segoe UI", 9, "bold"), padx=8, pady=5).pack(side=tk.LEFT)
            tk.Label(r_frame, text=b_val, width=col_widths[2], anchor="w",
                     bg=bg_col, fg="#38bdf8", font=("Segoe UI", 9, "bold"), padx=8, pady=5).pack(side=tk.LEFT)
            tk.Label(r_frame, text=delta_val, width=col_widths[3], anchor="w",
                     bg=bg_col, fg="#f59e0b", font=("Segoe UI", 9, "bold"), padx=8, pady=5).pack(side=tk.LEFT)

        # Analytical Summary Box
        summary_box = tk.Frame(content, bg=THEME["bg_card"], bd=1, relief=tk.SOLID)
        summary_box.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        tk.Label(
            summary_box, text="Theoretical & Engineering Analysis:",
            bg=THEME["bg_card"], fg=THEME["text_primary"],
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", padx=12, pady=(8, 4))

        if reduction_pct > 0:
            analysis_text = (
                f"• Exploration Efficiency: A* explored {astar_metrics.nodes_explored} nodes versus "
                f"BFS's {bfs_metrics.nodes_explored} nodes, achieving a {reduction_pct:.1f}% reduction in search space.\n"
                f"• Admissible Heuristic: Using {h_name} distance guarantees admissibility (h(n) ≤ h*(n)), "
                f"ensuring A* finds the optimal path without expanding wasteful concentric rings like BFS.\n"
                f"• Path Optimality: Both algorithms found paths of cost {astar_metrics.path_cost:.2f}, "
                f"demonstrating that A* achieves 100% path optimality with a fraction of node expansions."
            )
        else:
            analysis_text = (
                "• On small open topologies or trivial paths, both algorithms expand similar nodes.\n"
                "• Try creating obstacles, corridors, or using the 'U-Trap' preset to see A*'s heuristic advantage!"
            )

        tk.Label(
            summary_box, text=analysis_text,
            bg=THEME["bg_card"], fg=THEME["text_secondary"],
            font=("Segoe UI", 8), justify=tk.LEFT, wraplength=640
        ).pack(anchor="w", padx=12, pady=(0, 8))

        # Bottom Action Bar
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

        tk.Button(
            btn_bar, text="Show A* Visualization", command=show_astar_on_grid,
            bg="#2563eb", fg="#ffffff", font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT, padx=12, pady=6
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            btn_bar, text="Show BFS Visualization", command=show_bfs_on_grid,
            bg="#0284c7", fg="#ffffff", font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT, padx=12, pady=6
        ).pack(side=tk.LEFT, padx=6)

        tk.Button(
            btn_bar, text="Close", command=dlg.destroy,
            bg=THEME["bg_card"], fg=THEME["text_primary"],
            font=("Segoe UI", 9), relief=tk.FLAT, padx=16, pady=6
        ).pack(side=tk.RIGHT)
