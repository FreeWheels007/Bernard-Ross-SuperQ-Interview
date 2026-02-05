# Bernard-Ross-SuperQ-Interview
Bernard Ross SuperQ Interview
# CVRP MILP Solver (Interview Repo)

This repository contains a **Capacitated Vehicle Routing Problem (CVRP)** solver implemented as a **Mixed-Integer Linear Program (MILP)**. It reads a problem instance from JSON, builds and solves a MILP, and writes the resulting routes + per-vehicle loads to a machine-readable output (JSON), with optional plotting.

> Goal: demonstrate clean optimization modeling, careful constraint design (capacity + subtour elimination), and practical output handling (routes + loads) suitable for integration into a pipeline.

---

## Problem statement (CVRP)

Given:
- A depot (node `0`)
- Location nodes `1..n`
- Customer demands `demand_i`
- `V` vehicles with capacity `Q`
- A distance/cost matrix `distance_{ij}`

Find routes starting/ending at the depot such that:
- Every Location is visited exactly once
- Vehicle capacities are respected
- Total travel cost is minimized

---

## Repo structure

- `code/` — source code (model build, solve, utilities)
    - solver_main.py - main master program to read data, run gurobi solver, and write raw solution to file
    - results_reader_script.py - script to take raw solution into readable text/image (run after solver_main.py)
    - `Functions\` - functions run by solver_main.py
        - Data_reader.py - read/parse json input
        - MILP_solver_gurobipy.py - MILP algorithm with constraints/objectives
- `inputs/` — JSON problem instances  
- `outputs/` — solver results (JSON, optional plots)  


---

## Implementation overview

### Decision variables
Typical CVRP MILP formulation with:
- **Binary routing variables**  
    x_{ijv} = 1 for vehicle v going from i to j else 0
- **Continuous load variables (per vehicle)** 
    load_{jv} current load for vehicle v at j

### Objective
Minimize travel cost:
\[
\min \sum_{v}\sum_{i}\sum_{j} c_{ij}\,x_{ijv}
\]

### Core constraints (high level)
- **Visit each customer exactly once** (across all vehicles)
- **Flow conservation per vehicle**
- **Depot start/end constraints** (vehicles leave/return depot)
- **Capacity constraints** using load propagation demand

---

## Requirements

- Python 3.12.3 recommended
- A MILP-capable solver:
  - **Gurobi** (free via gurobipy)
- Common Python libs: `numpy`, `matplotlib`, etc.

> Conda environmnent.yml or pip requirements.txt files provided in `code\`

---

## Setup

### 1) Create environment (in code\)
```bash
cd code
conda env create -n superq -f environment.yml
conda activate superq
```

### 2) Run MILP solver
Adjust gap variable within code (set to 0.059)
```bash
python solver_main.py
```
Outputs a file within outputs\

### 3) Run results reader
Adjust gap variable within code (set to 0.059)
```bash
python results_reader_script.py
```
Prints to screen results and image

## Output sample to terminal (after running results_reader_script.py)

<img width="804" height="676" alt="image" src="https://github.com/user-attachments/assets/c532b93b-c4ec-4ee1-b7a6-cd494c606bce" />
