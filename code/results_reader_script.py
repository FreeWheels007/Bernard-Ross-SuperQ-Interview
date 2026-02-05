# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 09:58:32 2026

@author: berni
"""

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from Functions.Data_reader import read_data


KeyArc = Tuple[int, int, int]   # (i, j, v)
KeyLoad = Tuple[int, int]       # (node, v)


def print_pretty(
    json_path: str,
    number_locations: int,
    number_vehicles: int,
    gap: float,
    distance_matrix: Optional[np.ndarray] = None,
    demand_vector: Optional[List[float]] = None,
    depot: int = 0,
    eps: float = 0.5,
    plot: bool = True,
    plot_labels: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """
    Pretty-print CVRP solution from JSON + (optionally) plot routes using a 2D embedding
    computed from the distance matrix (no coordinates needed).

    Expected JSON formats supported:
      route_data: list-of-records [{"from":i,"to":j,"vehicle":v,"value":x}, ...]
              OR nested arrays route_data[i][j][v]
              OR dict with string keys "i,j,v"
      load_data:  list-of-records [{"node":j,"vehicle":v,"load":u}, ...]
              OR nested arrays load_data[j][v]
              OR dict with string keys "j,v"

    Returns:
      vehicle_info[v] = {"path": [...], "arcs":[(i,j),...], "loads":[...], "status": "..."}
    """

    # ------------------------
    # JSON -> maps
    # ------------------------
    def _build_maps_from_json(payload: Dict[str, Any]) -> tuple[Dict[KeyArc, float], Dict[KeyLoad, float]]:
        x_map: Dict[KeyArc, float] = {}
        load_map: Dict[KeyLoad, float] = {}

        route_data = payload.get("route_data")
        load_data = payload.get("load_data")

        # route_data
        if isinstance(route_data, list):
            if len(route_data) > 0 and isinstance(route_data[0], dict) and "from" in route_data[0]:
                for rec in route_data:
                    i = int(rec["from"])
                    j = int(rec["to"])
                    v = int(rec["vehicle"])
                    val = float(rec.get("value", rec.get("x", 0.0)))
                    x_map[(i, j, v)] = val
            else:
                # nested arrays route_data[i][j][v]
                for i, row in enumerate(route_data):
                    for j, col in enumerate(row):
                        for v, val in enumerate(col):
                            x_map[(i, j, v)] = float(val)

        elif isinstance(route_data, dict):
            # dict with key "i,j,v"
            for k, val in route_data.items():
                parts = str(k).split(",")
                if len(parts) != 3:
                    continue
                i, j, v = map(int, parts)
                x_map[(i, j, v)] = float(val)

        # load_data
        if isinstance(load_data, list):
            if len(load_data) > 0 and isinstance(load_data[0], dict) and "node" in load_data[0]:
                for rec in load_data:
                    node = int(rec["node"])
                    v = int(rec["vehicle"])
                    val = float(rec.get("load", rec.get("value", 0.0)))
                    load_map[(node, v)] = val
            else:
                # nested arrays load_data[node][v]
                for node, row in enumerate(load_data):
                    for v, val in enumerate(row):
                        load_map[(node, v)] = float(val)

        elif isinstance(load_data, dict):
            # dict with key "node,v"
            for k, val in load_data.items():
                parts = str(k).split(",")
                if len(parts) != 2:
                    continue
                node, v = map(int, parts)
                load_map[(node, v)] = float(val)

        return x_map, load_map

    # ------------------------
    # Extract ordered route for one vehicle
    # ------------------------
    def _extract_vehicle_route_from_maps(
        x_map: Dict[KeyArc, float],
        load_map: Dict[KeyLoad, float],
        n: int,
        v: int,
        depot: int = 0,
        eps: float = 0.5,
    ) -> tuple[List[int], List[Tuple[int, int]], List[float], str]:
        # succ[i] = j for this vehicle
        succ: Dict[int, int] = {}

        # Build successors using threshold; if none found for a node, later we fallback to argmax
        for i in range(n):
            best_j = None
            best_val = -1e100
            for j in range(n):
                if i == j:
                    continue
                val = x_map.get((i, j, v), 0.0)
                if val > best_val:
                    best_val = val
                    best_j = j
                if val > eps:
                    if i not in succ or val > x_map.get((i, succ[i], v), 0.0):
                        succ[i] = j

            # If depot has no threshold successor but has mass, set later by argmax.

        # Ensure depot has a successor if any arc weight exists
        if depot not in succ:
            best_j = None
            best_val = -1e100
            for j in range(n):
                if j == depot:
                    continue
                val = x_map.get((depot, j, v), 0.0)
                if val > best_val:
                    best_val = val
                    best_j = j
            if best_j is not None and best_val > 1e-12:
                succ[depot] = best_j

        if depot not in succ:
            ld0 = load_map.get((depot, v), 0.0)
            return [depot], [], [ld0], "No outgoing arc from depot (vehicle unused or bad extraction)."

        nodes = [depot]
        arcs: List[Tuple[int, int]] = []
        loads = [load_map.get((depot, v), 0.0)]

        cur = depot
        seen = {depot}

        for _ in range(n + 5):
            nxt = succ.get(cur)

            # Fallback to argmax outgoing if succ missing
            if nxt is None:
                best_j = None
                best_val = -1e100
                for j in range(n):
                    if j == cur:
                        continue
                    val = x_map.get((cur, j, v), 0.0)
                    if val > best_val:
                        best_val = val
                        best_j = j
                if best_j is None or best_val <= 1e-12:
                    break
                nxt = best_j

            arcs.append((cur, nxt))
            nodes.append(nxt)
            loads.append(load_map.get((nxt, v), float("nan")))

            cur = nxt
            if cur == depot:
                return nodes, arcs, loads, "OK (returned to depot)."
            if cur in seen:
                return nodes, arcs, loads, "Warning: cycle/subtour before returning to depot."
            seen.add(cur)

        return nodes, arcs, loads, "Warning: route ended without returning to depot (dead end/incomplete)."

    # ------------------------
    # Distance matrix -> 2D coords (classical MDS; sklearn not required)
    # ------------------------
    def _coords_from_distances_classical_mds(D: np.ndarray, seed: int = 0) -> np.ndarray:
        """
        Classical MDS (Torgerson): O(n^3) eigen-decomposition of double-centered squared distances.
        Works well for visualization; no sklearn dependency.
        """
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError("distance_matrix must be a square (n x n) array.")
        n = D.shape[0]
        D2 = D.astype(float) ** 2

        # Double centering: B = -1/2 * J D^2 J
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D2 @ J

        # Eigen-decomposition
        w, V = np.linalg.eigh(B)
        idx = np.argsort(w)[::-1]
        w = w[idx]
        V = V[:, idx]

        # Take top-2 positive eigenvalues
        w2 = np.maximum(w[:2], 0.0)
        coords = V[:, :2] * np.sqrt(w2)

        # Tiny random jitter if everything collapses (rare)
        if np.allclose(coords, 0.0):
            rng = np.random.default_rng(seed)
            coords = rng.normal(size=(n, 2))
        return coords

    # ------------------------
    # Plot routes
    # ------------------------
    def _plot_routes(gap: float, coords: np.ndarray, vehicle_info: Dict[int, Dict[str, Any]], depot: int = 0, labels: bool = True):
        fig, ax = plt.subplots(figsize=(9, 7))
    
        # nodes
        ax.scatter(coords[:, 0], coords[:, 1], zorder=3)
        # depot highlight
        ax.scatter([coords[depot, 0]], [coords[depot, 1]], s=160, zorder=4, label="Depot")
    
        if labels:
            for i in range(coords.shape[0]):
                ax.text(coords[i, 0], coords[i, 1], str(i))
    
        # one plotted line per vehicle => legend entries
        for v, info in vehicle_info.items():
            path = info["path"]
            if len(path) < 2:
                continue
    
            xs = [coords[n, 0] for n in path]
            ys = [coords[n, 1] for n in path]
    
            # Label for legend
            ax.plot(xs, ys, linewidth=2, marker="o", markersize=3, label=f"Vehicle {v}")
    
            # Add a route label near the middle segment (so you can see which is which quickly)
            mid_k = max(0, (len(path) - 1) // 2)
            mid_node = path[mid_k]
            ax.text(
                coords[mid_node, 0],
                coords[mid_node, 1],
                f"V{v}",
                fontsize=10,
                fontweight="bold",
                zorder=5
            )
    
        ax.set_title("CVRP Routes (High-level coordinate approximation)")
        ax.axis("equal")
        ax.grid(True)
    
        # Put legend outside so it doesn't cover the plot
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
        plt.tight_layout()
        plt.savefig(rf'..\outputs\plot_gap-{gap}.jpg', bbox_inches="tight")
        plt.show()

    # ------------------------
    # Main
    # ------------------------
    with open(json_path, "r") as f:
        payload = json.load(f)

    x_map, load_map = _build_maps_from_json(payload)

    print(f"For gap: {gap}; Minimum distance: {payload.get('Minimum distance')}")
    if "MIPGap" in payload:
        print(f"MIPGap: {payload.get('MIPGap')}")

    vehicle_info: Dict[int, Dict[str, Any]] = {}

    for v in range(number_vehicles):
        nodes, arcs, loads, status = _extract_vehicle_route_from_maps(
            x_map=x_map,
            load_map=load_map,
            n=number_locations,
            v=v,
            depot=depot,
            eps=eps,
        )

        vehicle_info[v] = {"path": nodes, "arcs": arcs, "loads": loads, "status": status}

        print(f"\n=== Vehicle {v} ===")
        print("Status:", status)

        if not arcs:
            print("Path:", str(depot))
            continue

        print("Path:", " -> ".join(map(str, nodes)))
        print("Stops (k, node, demand, load):")
        for k, node in enumerate(nodes):
            dem = 0.0
            if demand_vector is not None and node != depot:
                dem = float(demand_vector[node])

            ld = loads[k]
            dem_str = f"{dem:8.2f}" if demand_vector is not None else "   (n/a)"
            ld_str = f"{ld:8.2f}" if isinstance(ld, (int, float)) and not (isinstance(ld, float) and math.isnan(ld)) else "     nan"
            print(f"  {k:2d}: node={node:3d}, demand={dem_str}, load={ld_str}")

    # Plot if requested and we have distances
    if plot:
        if distance_matrix is None:
            print("\n[plot skipped] Provide distance_matrix to plot routes (you said you lack real coordinates).")
        else:
            D = np.asarray(distance_matrix, dtype=float)
            coords = _coords_from_distances_classical_mds(D)
            _plot_routes(gap, coords, vehicle_info, depot=depot, labels=plot_labels)

    return vehicle_info

if __name__ == '__main__':
    gap = 0.055
    distance_matrix, demand_vector, number_locations, number_vehicles, vehicle_capacity = read_data(r'..\inputs\cvrp_problem_data.json')
    
    vehicle_info = print_pretty(
        json_path=rf"..\outputs\results_gap-{gap}.json",
        number_locations=number_locations,
        number_vehicles=number_vehicles,
        gap=gap,
        distance_matrix=distance_matrix,   # numpy array (n x n)
        demand_vector=demand_vector,       # optional
        depot=0,
        eps=0.9,
        plot=True
    )

