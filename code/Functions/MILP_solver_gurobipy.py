# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 09:11:17 2026

@author: berni
"""

from gurobipy import Model, GRB, quicksum

def gurobipy_solver(distance_matrix, demand_vector, number_locations, number_vehicles, vehicle_capacity, gap):
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
    m.params.MIPGap = gap
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
        
    return m.ObjVal, route_data, load_data