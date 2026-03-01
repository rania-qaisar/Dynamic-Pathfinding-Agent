# Author : Rania Qaisar
"""
Dynamic Pathfinding Agent
=========================
Algorithms : A* Search, Greedy Best-First Search (GBFS)
Heuristics : Manhattan Distance, Euclidean Distance
Features   : Dynamic obstacle spawning, real-time re-planning,
             interactive map editor, random maze generator,
             animated visualisation with metrics dashboard.

Requirements: Python 3.x  (tkinter is included in the standard library)
Run: python pathfinding_agent.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import heapq
import random
import time
import math

# ─── Color Palette ────────────────────────────────────────────────────────────
COLORS = {
    "bg":        "#0a0d12",
    "panel":     "#0f1520",
    "border":    "#1e2d45",
    "text":      "#c8d8e8",
    "muted":     "#4a6080",
    "accent":    "#00d4ff",    #primary accent color
    "accent2":   "#ff6b35",
    # cell states
    "empty":     "#0b1320",
    "wall":      "#0d1e33",
    "start":     "#00d4ff",
    "goal":      "#ff6b35",
    "frontier":  "#ffd700",
    "visited":   "#1a2a44",
    "path":      "#39ff14",
    "agent":     "#ffffff",
}

# ─── Priority Queue (Min-Heap) ─────────────────────────────────────────────────
class PriorityQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0          # tie-breaker

    def push(self, priority, item):
        heapq.heappush(self._heap, (priority, self._counter, item))
        self._counter += 1

    def pop(self):
        _, _, item = heapq.heappop(self._heap)
        return item

    def __len__(self):
        return len(self._heap)


# ─── Search Algorithms ─────────────────────────────────────────────────────────
def manhattan(r1, c1, r2, c2):
    return abs(r1 - r2) + abs(c1 - c2)

def euclidean(r1, c1, r2, c2):
    return math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)

def get_neighbors(grid, rows, cols, r, c):
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 1:
            yield nr, nc

def search(grid, rows, cols, start, goal, algo, heuristic_fn):
    """
    Returns a generator that yields step dicts for animation.
    Step types:
      {'type': 'expand', 'node': (r,c), 'visited': set, 'frontier': set}
      {'type': 'done',   'path': [(r,c),...], 'visited': set}
      {'type': 'fail',   'visited': set}
    """
    def h(r, c):
        return heuristic_fn(r, c, goal[0], goal[1])

    pq = PriorityQueue()
    g_cost = {start: 0}
    parent = {start: None}
    visited = set()
    frontier_set = {start}

    f0 = h(*start) if algo == "gbfs" else (0 + h(*start))
    pq.push(f0, start)

    while len(pq):
        node = pq.pop()
        if node in visited:
            continue
        frontier_set.discard(node)
        visited.add(node)

        yield {"type": "expand", "node": node,
               "visited": set(visited), "frontier": set(frontier_set)}

        if node == goal:
            # Reconstruct path
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            yield {"type": "done", "path": path, "visited": set(visited)}
            return

        for nr, nc in get_neighbors(grid, rows, cols, *node):
            nb = (nr, nc)
            if nb in visited:
                continue
            new_g = g_cost[node] + 1
            if nb not in g_cost or new_g < g_cost[nb]:
                g_cost[nb] = new_g
                parent[nb] = node
                hn = h(nr, nc)
                fn = hn if algo == "gbfs" else (new_g + hn)
                pq.push(fn, nb)
                frontier_set.add(nb)

    yield {"type": "fail", "visited": set(visited)}


# ─── Main Application ──────────────────────────────────────────────────────────
class PathfindingApp:
    CELL = 24          # default cell pixel size (recalculated on resize)

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Dynamic Pathfinding Agent")
        self.root.configure(bg=COLORS["bg"])
        self.root.geometry("1280x780")
        self.root.minsize(900, 600)

        # ── Grid state ──
        self.rows = 20            #default grid rows
        self.cols = 30
        self.grid = None          # 0=empty, 1=wall
        self.start = (1, 1)
        self.goal  = (self.rows - 2, self.cols - 2)

        # ── Visualisation state ──
        self.visited_cells: set  = set()
        self.frontier_cells: set = set()
        self.path_cells: list    = []
        self.agent_pos           = None
        self.agent_path: list    = []
        self.agent_step: int     = 0

        # ── Algorithm / search state ──
        self.algo       = tk.StringVar(value="astar")
        self.heur       = tk.StringVar(value="manhattan")
        self.edit_mode  = tk.StringVar(value="wall")
        self.dynamic_on = tk.BooleanVar(value=False)
        self.density    = tk.IntVar(value=30)
        self.spawn_prob = tk.DoubleVar(value=3.0)
        self.viz_speed  = tk.IntVar(value=3)   # 1-5

        self.is_running   = False
        self.replan_count = 0
        self._search_gen  = None
        self._after_id    = None
        self._agent_id    = None
        self._dynamic_id  = None
        self._drawing     = False
        self._draw_value  = 1        # what to paint when dragging

        # ── Build UI ──
        self._build_ui()
        self._init_grid()
        self._generate_maze()       # start with a random maze
        self._draw_all()

        self.root.bind("<Configure>", self._on_resize)

    # ═══════════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ═══════════════════════════════════════════════════════════════════

    def _build_ui(self):
        # ── Root grid layout: left panel | canvas | right panel ──
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Header
        hdr = tk.Frame(self.root, bg=COLORS["panel"], height=46)
        hdr.grid(row=0, column=0, columnspan=3, sticky="ew")
        tk.Label(hdr, text="PATHFIND.EXE", font=("Courier", 16, "bold"),
                 bg=COLORS["panel"], fg=COLORS["accent"]).pack(side="left", padx=16, pady=8)
        tk.Label(hdr, text="Dynamic Grid Navigation Agent  |  A* & GBFS",
                 font=("Courier", 9), bg=COLORS["panel"],
                 fg=COLORS["muted"]).pack(side="left", pady=8)

        self.status_label = tk.Label(hdr, text="Ready",
                                     font=("Courier", 9), bg=COLORS["panel"],
                                     fg=COLORS["accent2"])
        self.status_label.pack(side="right", padx=16)

        # ── Left sidebar ──
        left = tk.Frame(self.root, bg=COLORS["panel"], width=230)
        left.grid(row=1, column=0, sticky="ns", padx=0, pady=0)
        left.grid_propagate(False)
        self._build_left(left)

        # ── Canvas frame ──
        canvas_frame = tk.Frame(self.root, bg=COLORS["bg"])
        canvas_frame.grid(row=1, column=1, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg=COLORS["bg"],
                                highlightthickness=1,
                                highlightbackground=COLORS["border"])
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        self.canvas.bind("<ButtonPress-1>",   self._on_mouse_press)
        self.canvas.bind("<B1-Motion>",       self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)

        # ── Right metrics panel ──
        right = tk.Frame(self.root, bg=COLORS["panel"], width=210)
        right.grid(row=1, column=2, sticky="ns")
        right.grid_propagate(False)
        self._build_right(right)

    # ─ Left sidebar ──────────────────────────────────────────────────────────

    def _build_left(self, parent):
        pad = {"padx": 10, "pady": 3}

        def section(text):
            tk.Label(parent, text=f"── {text} ──", font=("Courier", 8, "bold"),
                     bg=COLORS["panel"], fg=COLORS["accent"]).pack(fill="x", **pad)

        def slider_row(parent, text, var, from_, to, cmd=None):
            row = tk.Frame(parent, bg=COLORS["panel"])
            row.pack(fill="x", padx=10, pady=1)
            tk.Label(row, text=text, font=("Courier", 8), width=18, anchor="w",
                     bg=COLORS["panel"], fg=COLORS["muted"]).pack(side="left")
            val_lbl = tk.Label(row, textvariable=var, font=("Courier", 8),
                               bg=COLORS["panel"], fg=COLORS["accent"], width=4)
            val_lbl.pack(side="right")
            s = ttk.Scale(row, variable=var, from_=from_, to=to,
                          orient="horizontal", command=cmd)
            s.pack(side="left", fill="x", expand=True)
            return s

        def radio_row(parent, text, var, value, color=None):
            f = tk.Frame(parent, bg=COLORS["panel"])
            f.pack(fill="x", padx=10, pady=1)
            rb = tk.Radiobutton(f, text=text, variable=var, value=value,
                                bg=COLORS["panel"], fg=color or COLORS["text"],
                                selectcolor=COLORS["bg"],
                                activebackground=COLORS["panel"],
                                font=("Courier", 9))
            rb.pack(side="left")

        def btn(parent, text, cmd, color=COLORS["accent"]):
            b = tk.Button(parent, text=text, command=cmd,
                          bg=COLORS["bg"], fg=color,
                          relief="flat", bd=1,
                          font=("Courier", 9, "bold"),
                          activebackground=COLORS["border"],
                          activeforeground=color,
                          cursor="hand2", padx=6, pady=4)
            return b

        # Grid dimensions
        section("GRID SIZE")
        self.rows_var = tk.IntVar(value=self.rows)
        self.cols_var = tk.IntVar(value=self.cols)
        slider_row(parent, "Rows", self.rows_var, 5, 50)
        slider_row(parent, "Cols", self.cols_var, 5, 60)

        f_apply = tk.Frame(parent, bg=COLORS["panel"])
        f_apply.pack(fill="x", padx=10, pady=4)
        btn(f_apply, "⟳  Apply Grid", self._apply_grid).pack(fill="x")

        # Obstacle density
        section("MAZE DENSITY")
        slider_row(parent, "Wall %", self.density, 0, 60)
        f_maze = tk.Frame(parent, bg=COLORS["panel"])
        f_maze.pack(fill="x", padx=10, pady=4)
        btn(f_maze, "⟳  Generate Maze", self._generate_maze,
            COLORS["accent2"]).pack(fill="x")

        # Algorithm
        section("ALGORITHM")
        radio_row(parent, "A*  (f = g + h)", self.algo, "astar",   COLORS["accent"])
        radio_row(parent, "GBFS  (f = h)",   self.algo, "gbfs",    COLORS["accent2"])

        # Heuristic
        section("HEURISTIC")
        radio_row(parent, "Manhattan  |x₁−x₂|+|y₁−y₂|", self.heur, "manhattan")
        radio_row(parent, "Euclidean  √((x₁−x₂)²+(y₁−y₂)²)", self.heur, "euclidean")

        # Edit mode
        section("EDIT MODE")
        modes = [("🧱 Draw Wall", "wall", COLORS["text"]),
                 ("🔵 Set Start", "start", COLORS["start"]),
                 ("🟠 Set Goal",  "goal",  COLORS["goal"])]
        for lbl, val, col in modes:
            radio_row(parent, lbl, self.edit_mode, val, col)

        # Dynamic obstacles
        section("DYNAMIC MODE")
        chk_f = tk.Frame(parent, bg=COLORS["panel"])
        chk_f.pack(fill="x", padx=10, pady=2)
        tk.Checkbutton(chk_f, text="Enable obstacle spawning",
                       variable=self.dynamic_on,
                       bg=COLORS["panel"], fg=COLORS["text"],
                       selectcolor=COLORS["bg"],
                       activebackground=COLORS["panel"],
                       font=("Courier", 9)).pack(side="left")
        slider_row(parent, "Spawn %", self.spawn_prob, 1, 15)

        # Viz speed
        section("VIZ SPEED")
        slider_row(parent, "Speed (1–5)", self.viz_speed, 1, 5)

        # Action buttons
        section("ACTIONS")
        f_btns = tk.Frame(parent, bg=COLORS["panel"])
        f_btns.pack(fill="x", padx=10, pady=4)

        btn(f_btns, "▶  RUN",       self._start_search, COLORS["accent"]).pack(fill="x", pady=2)
        btn(f_btns, "■  STOP",      self._stop,         COLORS["accent2"]).pack(fill="x", pady=2)
        btn(f_btns, "✖  Clear Path",self._clear_path,   COLORS["muted"]).pack(fill="x", pady=2)
        btn(f_btns, "⚠  Reset All", self._reset_all,    "#ff3366").pack(fill="x", pady=2)

    # ─ Right metrics panel ────────────────────────────────────────────────────

    def _build_right(self, parent):
        def section(text):
            tk.Label(parent, text=f"── {text} ──", font=("Courier", 8, "bold"),
                     bg=COLORS["panel"], fg=COLORS["accent"]).pack(fill="x", padx=10, pady=4)

        def metric_card(parent, label, attr, color):
            card = tk.Frame(parent, bg=COLORS["bg"], relief="flat", bd=1)
            card.pack(fill="x", padx=10, pady=3)
            tk.Label(card, text=label, font=("Courier", 8),
                     bg=COLORS["bg"], fg=COLORS["muted"]).pack(anchor="w", padx=8, pady=(6,0))
            lbl = tk.Label(card, text="0", font=("Courier", 22, "bold"),
                           bg=COLORS["bg"], fg=color)
            lbl.pack(anchor="w", padx=8)
            setattr(self, attr, lbl)
            return lbl

        section("METRICS")
        metric_card(parent, "NODES VISITED",    "_lbl_nodes",   COLORS["accent"])
        metric_card(parent, "PATH COST",        "_lbl_cost",    COLORS["path"])
        metric_card(parent, "EXEC TIME (ms)",   "_lbl_time",    COLORS["frontier"])
        metric_card(parent, "RE-PLANS",         "_lbl_replans", "#a855f7")

        section("LEGEND")
        legends = [
            (COLORS["start"],    "Start Node"),
            (COLORS["goal"],     "Goal Node"),
            (COLORS["wall"],     "Wall"),
            (COLORS["frontier"], "Frontier"),
            (COLORS["visited"],  "Visited"),
            (COLORS["path"],     "Final Path"),
            (COLORS["agent"],    "Agent"),
        ]
        for color, label in legends:
            row = tk.Frame(parent, bg=COLORS["panel"])
            row.pack(fill="x", padx=12, pady=1)
            sq = tk.Canvas(row, width=14, height=14, bg=color,
                           highlightthickness=1, highlightbackground=COLORS["border"])
            sq.pack(side="left")
            tk.Label(row, text=f"  {label}", font=("Courier", 8),
                     bg=COLORS["panel"], fg=COLORS["muted"]).pack(side="left")

        section("EVENT LOG")
        log_frame = tk.Frame(parent, bg=COLORS["panel"])
        log_frame.pack(fill="both", expand=True, padx=10, pady=4)
        scroll = tk.Scrollbar(log_frame, bg=COLORS["bg"])
        scroll.pack(side="right", fill="y")
        self._log_box = tk.Text(log_frame, height=10,
                                bg="#050810", fg=COLORS["muted"],
                                font=("Courier", 8),
                                relief="flat", bd=0,
                                yscrollcommand=scroll.set,
                                state="disabled")
        self._log_box.pack(fill="both", expand=True)
        scroll.config(command=self._log_box.yview)

        # colour tags
        self._log_box.tag_config("info",    foreground=COLORS["accent"])
        self._log_box.tag_config("warn",    foreground=COLORS["frontier"])
        self._log_box.tag_config("success", foreground=COLORS["path"])
        self._log_box.tag_config("error",   foreground="#ff3366")
        self._log_box.tag_config("default", foreground=COLORS["muted"])

    # ═══════════════════════════════════════════════════════════════════
    # GRID & CANVAS HELPERS
    # ═══════════════════════════════════════════════════════════════════

    def _init_grid(self):
        self.grid = [[0] * self.cols for _ in range(self.rows)]

    def _cell_size(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 2 or h < 2:
            return self.CELL
        return max(6, min(w // self.cols, h // self.rows))

    def _canvas_to_cell(self, x, y):
        cs = self._cell_size()
        return y // cs, x // cs   # (row, col)

    def _draw_all(self):
        self.canvas.delete("all")
        cs = self._cell_size()
        for r in range(self.rows):
            for c in range(self.cols):
                self._draw_cell(r, c, cs)
        # Draw agent on top
        if self.agent_pos:
            r, c = self.agent_pos
            x, y = c * cs, r * cs
            margin = max(2, cs // 5)
            self.canvas.create_oval(
                x + margin, y + margin,
                x + cs - margin, y + cs - margin,
                fill=COLORS["agent"], outline=COLORS["accent"], width=2,
                tags="agent"
            )

    def _draw_cell(self, r, c, cs=None):
        if cs is None:
            cs = self._cell_size()
        x1, y1 = c * cs, r * cs
        x2, y2 = x1 + cs, y1 + cs
        cell_key = (r, c)

        if cell_key == self.start:
            color = COLORS["start"]
            text, tcolor = "S", "#000000"
        elif cell_key == self.goal:
            color = COLORS["goal"]
            text, tcolor = "G", "#000000"
        elif self.grid[r][c] == 1:
            color = COLORS["wall"]
            text, tcolor = None, None
        elif cell_key in [(kr, kc) for kr, kc in
                          [divmod(k, 1000) for k in self.path_cells]]:
            color = COLORS["path"]
            text, tcolor = None, None
        elif cell_key in self.visited_cells:
            color = COLORS["visited"]
            text, tcolor = None, None
        elif cell_key in self.frontier_cells:
            color = COLORS["frontier"]
            text, tcolor = None, None
        else:
            color = COLORS["empty"]
            text, tcolor = None, None

        self.canvas.create_rectangle(x1, y1, x2, y2,
                                      fill=color,
                                      outline=COLORS["border"],
                                      width=0 if cs < 10 else 1)
        if text and cs >= 12:
            self.canvas.create_text(x1 + cs // 2, y1 + cs // 2,
                                     text=text, fill=tcolor,
                                     font=("Courier", max(7, cs // 2 - 1), "bold"))

    # ── Uses flat sets for faster lookup ────────────────────────────────────
    # We store visited/frontier as sets of (r,c) tuples

    # ═══════════════════════════════════════════════════════════════════
    # MAZE GENERATION
    # ═══════════════════════════════════════════════════════════════════

    def _generate_maze(self):
        self._stop()
        self._init_grid()
        density = self.density.get() / 100.0
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in (self.start, self.goal):
                    continue
                self.grid[r][c] = 1 if random.random() < density else 0
        self.visited_cells  = set()
        self.frontier_cells = set()
        self.path_cells     = []
        self.agent_pos      = None
        self._update_metrics(0, 0, 0)
        self._draw_all()
        self._log("info", f"Maze generated  ({self.density.get()}% walls)")

    # ═══════════════════════════════════════════════════════════════════
    # MOUSE INTERACTION
    # ═══════════════════════════════════════════════════════════════════

    def _on_mouse_press(self, event):
        if self.is_running:
            return
        self._drawing = True
        r, c = self._canvas_to_cell(event.x, event.y)
        if not self._valid(r, c):
            return
        mode = self.edit_mode.get()
        if mode == "start":
            if (r, c) != self.goal:
                self.start = (r, c)
                self.grid[r][c] = 0
                self._draw_all()
        elif mode == "goal":
            if (r, c) != self.start:
                self.goal = (r, c)
                self.grid[r][c] = 0
                self._draw_all()
        elif mode == "wall":
            self._draw_value = 0 if self.grid[r][c] == 1 else 1
            self._toggle_wall(r, c)

    def _on_mouse_drag(self, event):
        if not self._drawing or self.edit_mode.get() != "wall":
            return
        r, c = self._canvas_to_cell(event.x, event.y)
        if self._valid(r, c):
            self._toggle_wall(r, c, force=self._draw_value)

    def _on_mouse_release(self, event):
        self._drawing = False

    def _toggle_wall(self, r, c, force=None):
        if (r, c) in (self.start, self.goal):
            return
        val = force if force is not None else (0 if self.grid[r][c] == 1 else 1)
        self.grid[r][c] = val
        self._draw_cell(r, c)

    def _valid(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    # ═══════════════════════════════════════════════════════════════════
    # SEARCH CONTROL
    # ═══════════════════════════════════════════════════════════════════

    def _start_search(self, from_pos=None):
        self._stop_timers()
        self.visited_cells  = set()
        self.frontier_cells = set()
        self.path_cells     = []
        self.agent_pos      = None
        self.is_running     = True
        self._draw_all()

        start_r, start_c = from_pos if from_pos else self.start
        algo = self.algo.get()
        hfn  = manhattan if self.heur.get() == "manhattan" else euclidean

        self._search_gen = search(
            self.grid, self.rows, self.cols,
            (start_r, start_c), self.goal,
            algo, hfn
        )
        self._t0 = time.perf_counter()
        self._nodes_visited = 0
        self._log("info", f"Running {algo.upper()} / {self.heur.get()}")
        self._set_status(f"▶ {algo.upper()} searching…")
        self._step_search()

    def _step_search(self):
        if not self.is_running or self._search_gen is None:
            return

        # How many steps per tick?
        speed = max(1, int(self.viz_speed.get()))
        delay_map = {1: 150, 2: 80, 3: 30, 4: 8, 5: 1}
        delay = delay_map.get(speed, 30)
        steps_per_tick = 1 if speed < 4 else 10

        for _ in range(steps_per_tick):
            try:
                step = next(self._search_gen)
            except StopIteration:
                break

            if step["type"] == "expand":
                self.visited_cells  = step["visited"]
                self.frontier_cells = step["frontier"]
                self._nodes_visited = len(self.visited_cells)
                self._lbl_nodes.config(text=str(self._nodes_visited))

            elif step["type"] == "done":
                elapsed_ms = (time.perf_counter() - self._t0) * 1000
                self.visited_cells  = step["visited"]
                self.frontier_cells = set()
                path_tuples = step["path"]
                # encode as flat ints for backward compat
                self.path_cells = [r * 1000 + c for r, c in path_tuples]
                self._draw_all()
                cost = len(path_tuples) - 1
                self._update_metrics(self._nodes_visited, cost, round(elapsed_ms, 1))
                self._log("success",
                          f"Path found  cost={cost}  nodes={self._nodes_visited}"
                          f"  time={round(elapsed_ms,1)}ms")
                self._set_status("Path found — walking…")
                # Start agent animation
                self.agent_path = path_tuples
                self.agent_step = 0
                self._walk_agent()
                return

            elif step["type"] == "fail":
                self.visited_cells  = step["visited"]
                self.frontier_cells = set()
                self._draw_all()
                self._log("error", "No path found!")
                self._set_status("✖ No path found")
                self._end_search()
                return

        self._draw_all()
        self._after_id = self.root.after(delay, self._step_search)

    # ─── Agent animation ─────────────────────────────────────────────────────

    def _walk_agent(self):
        if not self.is_running:
            return
        if self.agent_step >= len(self.agent_path):
            self.agent_pos = None
            self._end_search()
            self._set_status("✔ Goal reached!")
            self._log("success", "Agent reached goal ✓")
            return

        self.agent_pos = self.agent_path[self.agent_step]
        self.agent_step += 1
        self._draw_all()

        speed = max(1, int(self.viz_speed.get()))
        delay_map = {1: 200, 2: 120, 3: 60, 4: 25, 5: 8}
        delay = delay_map.get(speed, 60)

        self._agent_id = self.root.after(delay, self._walk_agent)

        # Dynamic obstacle spawning (runs alongside walk)
        if self.dynamic_on.get():
            self._maybe_spawn_obstacle()

    def _maybe_spawn_obstacle(self):
        prob = self.spawn_prob.get() / 100.0
        if random.random() > prob:
            return

        # Pick a random empty cell (not start, goal, or agent)
        candidates = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.grid[r][c] == 0
            and (r, c) != self.start
            and (r, c) != self.goal
            and (r, c) != self.agent_pos
        ]
        if not candidates:
            return

        nr, nc = random.choice(candidates)
        self.grid[nr][nc] = 1
        self._log("warn", f"Obstacle spawned @ ({nr},{nc})")

        # Check if obstacle is on remaining path
        remaining = self.agent_path[self.agent_step:]
        if (nr, nc) in remaining:
            self._log("warn", "Obstacle on path — RE-PLANNING…")
            self.replan_count += 1
            self._lbl_replans.config(text=str(self.replan_count))
            # Cancel current walk and replan from agent position
            if self._agent_id:
                self.root.after_cancel(self._agent_id)
                self._agent_id = None
            current = self.agent_pos or self.start
            self._start_search(from_pos=current)

    # ═══════════════════════════════════════════════════════════════════
    # UTILITY
    # ═══════════════════════════════════════════════════════════════════

    def _end_search(self):
        self.is_running = False
        self._search_gen = None
        document_id = None

    def _stop(self):
        self._stop_timers()
        self.is_running = False
        self._search_gen = None
        self._set_status("Stopped")

    def _stop_timers(self):
        for attr in ("_after_id", "_agent_id", "_dynamic_id"):
            tid = getattr(self, attr, None)
            if tid:
                self.root.after_cancel(tid)
                setattr(self, attr, None)

    def _clear_path(self):
        self._stop()
        self.visited_cells  = set()
        self.frontier_cells = set()
        self.path_cells     = []
        self.agent_pos      = None
        self.replan_count   = 0
        self._update_metrics(0, 0, 0)
        self._lbl_replans.config(text="0")
        self._draw_all()
        self._set_status("Path cleared")

    def _reset_all(self):
        self._stop()
        self.rows = max(5, min(50, int(self.rows_var.get())))
        self.cols = max(5, min(60, int(self.cols_var.get())))
        self.start = (1, 1)
        self.goal  = (self.rows - 2, self.cols - 2)
        self._init_grid()
        self.visited_cells  = set()
        self.frontier_cells = set()
        self.path_cells     = []
        self.agent_pos      = None
        self.replan_count   = 0
        self._update_metrics(0, 0, 0)
        self._lbl_replans.config(text="0")
        self._draw_all()
        self._log("info", f"Grid reset  {self.rows}×{self.cols}")
        self._set_status("Grid reset")

    def _apply_grid(self):
        self._reset_all()

    def _update_metrics(self, nodes, cost, time_ms):
        self._lbl_nodes.config(text=str(nodes))
        self._lbl_cost.config(text=str(cost))
        self._lbl_time.config(text=str(time_ms))

    def _set_status(self, msg):
        self.status_label.config(text=msg)

    def _log(self, tag, msg):
        self._log_box.config(state="normal")
        ts = time.strftime("%H:%M:%S")
        self._log_box.insert("end", f"[{ts}] {msg}\n", tag)
        self._log_box.see("end")
        self._log_box.config(state="disabled")

    def _on_resize(self, event):
        self._draw_all()

    # ─── path_cells compatibility: stored as int keys ──────────────────────
    # visited_cells stored as (r,c) tuples for fast lookup in _draw_cell

    # Override _draw_cell to use tuple path lookup
    # (re-define above logic cleanly)
    def _draw_cell(self, r, c, cs=None):
        if cs is None:
            cs = self._cell_size()
        x1, y1 = c * cs, r * cs
        x2, y2 = x1 + cs, y1 + cs
        cell_key = (r, c)
        pk = r * 1000 + c

        if cell_key == self.start:
            color = COLORS["start"]
            text, tcolor = "S", "#000000"
        elif cell_key == self.goal:
            color = COLORS["goal"]
            text, tcolor = "G", "#000000"
        elif self.grid[r][c] == 1:
            color = COLORS["wall"]
            text, tcolor = None, None
        elif pk in self.path_cells:
            color = COLORS["path"]
            text, tcolor = None, None
        elif cell_key in self.visited_cells:
            color = COLORS["visited"]
            text, tcolor = None, None
        elif cell_key in self.frontier_cells:
            color = COLORS["frontier"]
            text, tcolor = None, None
        else:
            color = COLORS["empty"]
            text, tcolor = None, None

        self.canvas.create_rectangle(x1, y1, x2, y2,
                                      fill=color,
                                      outline=COLORS["border"],
                                      width=0 if cs < 10 else 1)
        if text and cs >= 12:
            self.canvas.create_text(x1 + cs // 2, y1 + cs // 2,
                                     text=text, fill=tcolor,
                                     font=("Courier", max(7, cs // 2 - 1), "bold"))


# ─── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()

    # Style ttk widgets
    style = ttk.Style()
    style.theme_use("default")
    style.configure("Horizontal.TScale",
                    background=COLORS["panel"],
                    troughcolor=COLORS["border"],
                    sliderthickness=14)

    app = PathfindingApp(root)

    root.mainloop()


