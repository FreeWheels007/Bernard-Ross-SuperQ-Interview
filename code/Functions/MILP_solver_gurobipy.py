# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 09:11:17 2026

@author: Bernard

MILP solver using gurobipy to solve CVRP
"""

from gurobipy import Model, GRB, quicksum

def gurobipy_solver(distance_matrix, demand_vector, number_locations, number_vehicles, vehicle_capacity, gap):
    """
    uses gurobipy to solve MILP CVRP
    Requires:
        distance matrix
        demand vector
        number of locations (including depot)
        number of vehicles
        gap vale
    
    Returns:
        Minimum distance
        route data (all binary values representing paths between locations) in list
        load data (all load for each vehicle at each location) in list
    """
    
    m = Model('superq') # initialize model
    
    # set up indices that we will use to represent gurobi decision variables*******************
    # x_ijv are valid matrix elements representing if vehicle v travels from location i to j
    # j cannot equal i to avoid trivial solutions (vehicle staying still)
    x_ijv = [(i, j, v) for i in range(number_locations) for j in range(number_locations) if j != i for v in range(number_vehicles)]
    
    # load_jv are valid matrix elements tracking the cumulative load on vehicle v as it reaches location j
    load_jv = [(j, v) for j in range(number_locations) for v in range(number_vehicles)]
    
    # create the gurobipy decision variables, using valid elements
    x = m.addVars(x_ijv, vtype=GRB.BINARY)
    load = m.addVars(load_jv, vtype=GRB.CONTINUOUS)
    
    # create objective function, minimizing total distance travelled across all vehicles
    m.ModelSense = GRB.MINIMIZE
    m.setObjective(quicksum(x[i, j, v]*distance_matrix[i, j] for i, j, v in x_ijv))
    
    # CONSTRAINTS
    
    # Vehicle path logic *******************************************************************
    # Each location must be be ARRIVED at by a vehicle
    m.addConstrs(quicksum(x[i, j, v] for i in range(number_locations) if i != j for v in range(number_vehicles)) == 1 
                 for j in range(1, number_locations))
    
    # Each location must be DEPARTED by a vehicle
    m.addConstrs(quicksum(x[i, j, v] for j in range(number_locations) if j != i for v in range(number_vehicles)) == 1
                 for i in range(1, number_locations))
    
    # Every vehicle can only ever leave from a location it is add - path consistancy
    m.addConstrs(quicksum(x[i, j, v] for j in range(number_locations) if j != i) == quicksum(x[j, i, v] for j in range(number_locations) if j != i)
                 for i in range(1, number_locations) for v in range(number_vehicles))
    
    # Each vehicle must start and end its path at depot
    m.addConstrs(quicksum(x[0, j, v] for j in range(1, number_locations)) == 1 for v in range(number_vehicles))
    m.addConstrs(quicksum(x[j, 0, v] for j in range(1, number_locations)) == 1 for v in range(number_vehicles))
    
    # Vehicle capacity logic **************************************************************
    # Each vehicle must start empty at depot
    m.addConstrs(load[0, v] == 0 for v in range(number_vehicles))
    
    # If vehicle travels from location i to j, then the its cumulative load at j must be its old load at i
    # plus what it collects at j
    m.addConstrs((x[i, j, v] == 1) >> (load[j, v] == load[i, v] + demand_vector[j]) 
                 for i, j, v in x_ijv if i !=0 and j != 0)
    
    # each vehicle's load at j cannot be less than the demand at j, provided it came from any i to j
    m.addConstrs(load[j, v] >= demand_vector[j]*quicksum(x[i, j, v] for i in range(number_locations) if j != i) 
                 for j in range(1, number_locations) for v in range(number_vehicles))
    
    # Each vehicle's load may not excede capacity
    m.addConstrs(load[j, v] <= vehicle_capacity for j in range(1, number_locations) for v in range(number_vehicles))
    
    # *************************************************************************************************
    
    # Run solver at gap accuracy
    print(m.printStats())
    m.params.MIPGap = gap
    m.optimize()

    # repackage solution in python objects
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