# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 09:11:17 2026

@author: berni
"""

import json
import numpy as np
from gurobipy import Model, GRB, quicksum

with open(r'.\inputs\cvrp_problem_data.json', 'r') as fp:
    input_data = json.load(fp)
    
distance_matrix = np.array(input_data['distance_matrix'])
demand_vector = np.array(input_data['demands'])
print(sum(demand_vector))

number_vehicles = input_data['vehicles']['count']
number_locations = input_data['nodes']['total']
vehicle_capacity = input_data['vehicles']['capacity_per_vehicle']

m = Model('superq')

x_ijv = [(i, j, v) for i in range(number_locations) for j in range(number_locations) if j != i for v in range(number_vehicles)]
load_jv = [(j, v) for j in range(number_locations) for v in range(number_vehicles)]

x = m.addVars(x_ijv, vtype=GRB.BINARY)
load = m.addVars(load_jv, vtype=GRB.CONTINUOUS)

m.ModelSense = GRB.MINIMIZE

m.setObjective(quicksum(x[i, j, v]*distance_matrix[i, j] for i, j, v in x_ijv))


m.addConstrs(quicksum(x[i, j, v] for i in range(number_locations) if i != j for v in range(number_vehicles)) == 1 
             for j in range(1, number_locations))

m.addConstrs(quicksum(x[i, j, v] for j in range(number_locations) if j != i for v in range(number_vehicles)) == 1
             for i in range(1, number_locations))

# Vehicle flow consistency – When a vehicle arrives at a location, it can only leave from same location
m.addConstrs(quicksum(x[i, j, v] for j in range(number_locations) if j != i) == quicksum(x[j, i, v] for j in range(number_locations) if j != i)
             for i in range(1, number_locations) for v in range(number_vehicles))

m.addConstrs(quicksum(x[0, j, v] for j in range(1, number_locations)) == 1 for v in range(number_vehicles))
m.addConstrs(quicksum(x[j, 0, v] for j in range(1, number_locations)) == 1 for v in range(number_vehicles))

m.addConstrs(load[0, v] == 0 for v in range(number_vehicles))

m.addConstrs((x[i, j, v] == 1) >> (load[j, v] == load[i, v] + demand_vector[j]) 
             for i, j, v in x_ijv if i !=0 and j != 0)

m.addConstrs(load[j, v] >= demand_vector[j]*quicksum(x[i, j, v] for i in range(number_locations) if j != i) 
             for j in range(1, number_locations) for v in range(number_vehicles))
m.addConstrs(load[j, v] <= vehicle_capacity for j in range(1, number_locations) for v in range(number_vehicles))

print(m.printStats())
m.params.MIPGap = 0.07
m.optimize()

route_data = []
for (i, j, v), var in x.items():
    route_data.append({
        "from": i,
        "to": j,
        "vehicle": v,
        "value": var.X
    })

load_data = []
for (j, v), var in load.items():
    load_data.append({
        "node": j,
        "vehicle": v,
        "load": var.X
    })
    
with open(r'.\outputs\results.json', 'w') as fp:
    json.dump({
        'Minimum distance': m.ObjVal,
        'route_data': route_data,
        'load_data': load_data
    }, fp, indent=4)

# --- AFTER m.optimize() ---

def extract_vehicle_route(x, load, number_locations, v, depot=0, eps=0.5):
    """
    Returns (route_nodes, route_arcs, route_loads)
      - route_nodes: [0, ..., 0]
      - route_arcs:  [(i,j), ...]
      - route_loads: [load at node in same order as route_nodes]
    """
    # Build successor map for this vehicle: i -> j where x[i,j,v]=1
    succ = {}
    for i in range(number_locations):
        for j in range(number_locations):
            if i == j:
                continue
            # x is a tupledict keyed by (i,j,v)
            if (i, j, v) in x and x[i, j, v].X > eps:
                succ[i] = j

    if depot not in succ:
        return [depot], [], [load[depot, v].X if (depot, v) in load else 0.0]

    route_nodes = [depot]
    route_arcs = []
    route_loads = [load[depot, v].X]

    cur = depot
    seen = set([depot])

    # Walk until we return to depot (or detect a subtour / dead end)
    for _ in range(number_locations + 5):
        nxt = succ.get(cur, None)
        if nxt is None:
            break

        route_arcs.append((cur, nxt))
        route_nodes.append(nxt)

        # Load at the node (depot load exists; customer loads exist)
        if (nxt, v) in load:
            route_loads.append(load[nxt, v].X)
        else:
            route_loads.append(float("nan"))

        cur = nxt

        if cur == depot:
            break

        # safety: if we loop without returning to depot
        if cur in seen:
            # indicates a subtour or repeated node (shouldn't happen if model is correct)
            break
        seen.add(cur)

    return route_nodes, route_arcs, route_loads


def pretty_print_routes(x, load, demand_vector, number_locations, number_vehicles, depot=0):
    total_dist = 0.0

    for v in range(number_vehicles):
        nodes, arcs, loads = extract_vehicle_route(x, load, number_locations, v, depot=depot)

        print(f"\n=== Vehicle {v} ===")
        if len(arcs) == 0:
            print("No route found (vehicle unused or bad extraction).")
            continue

        # Print path as 0 -> a -> b -> 0
        print("Path:", " -> ".join(map(str, nodes)))

        # Print per-stop details (skip the first depot)
        print("Stops (node, demand, load):")
        for k, node in enumerate(nodes):
            dem = demand_vector[node] if node != depot else 0
            ld = loads[k]
            print(f"  {k:2d}: node={node:3d}, demand={dem:6.2f}, load={ld:8.2f}")

        # Optional: compute distance of this vehicle using your distance_matrix in outer scope
        try:
            veh_dist = sum(distance_matrix[i, j] for (i, j) in arcs)
            total_dist += veh_dist
            print(f"Route distance: {veh_dist:.3f}")
        except NameError:
            pass

    try:
        print(f"\nTotal distance (from printed arcs): {total_dist:.3f}")
        print(f"Objective value (model): {m.ObjVal:.3f}")
    except Exception:
        pass


pretty_print_routes(x, load, demand_vector, number_locations, number_vehicles)

