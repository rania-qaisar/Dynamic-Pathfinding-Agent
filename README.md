# Dynamic Pathfinding Agent

A grid-based pathfinding visualizer implementing A* and Greedy Best-First Search (GBFS) with dynamic obstacle spawning and real-time replanning.

## Features
- A* Search and Greedy Best-First Search (GBFS)
- Manhattan and Euclidean heuristics
- Interactive map editor (draw/erase walls)
- Random maze generation with configurable density
- Dynamic obstacle spawning with real-time replanning
- Animated visualization with metrics dashboard

## Requirements
- Python 3.x
- Tkinter (included in Python standard library)

## How to Run
1. Clone the repository or download the file
2. Open terminal and navigate to the folder
3. Run:
```
python pathfinding_agent.py
```

## How to Use
- **Draw walls** by clicking/dragging on the grid
- Select **Start** or **Goal** mode to reposition nodes
- Choose **Algorithm** (A* or GBFS) and **Heuristic**
- Click **RUN** to start the search
- Enable **Dynamic Mode** for real-time obstacle spawning
- Use **Generate Maze** for a random map

## Algorithms
| Algorithm | Function |
|-----------|----------|
| A* | f(n) = g(n) + h(n) |
| GBFS | f(n) = h(n) |

## Heuristics
| Heuristic | Formula |
|-----------|---------|
| Manhattan | \|x₁−x₂\| + \|y₁−y₂\| |
| Euclidean | √((x₁−x₂)² + (y₁−y₂)²) |
```
## Author
**Rania Qaisar**
