# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 09:11:17 2026

@author: Bernard

Main script to coordinate reading input data, then running (free) Gurobi solver on system constrains,
then writing all data to file 
"""

import json
from Functions.Data_reader import read_data
from Functions.MILP_solver_gurobipy import gurobipy_solver

if __name__ == '__main__':
    # read in json data into numpy arrays amd scalar values
    distance_matrix, demand_vector, number_locations, number_vehicles, vehicle_capacity = read_data(r'..\inputs\cvrp_problem_data.json')
    
    # define gap limit, which represents how accurate final solution will be
    # smaller values take far longer runtimes, but are more accurate (closer to finding the min distance routes)
    gap = 0.059
    min_distance, routes, loads = gurobipy_solver(distance_matrix, demand_vector, number_locations, number_vehicles, vehicle_capacity, gap)
    
    # once solution is found, write solution to json file for storage, 
    #later processing/beautifying in results_reader_script.py
    out_path = rf'..\outputs\results_gap-{gap}.json'
    with open(out_path, 'w') as fp:
        json.dump({
            'Minimum distance': min_distance,
            'route_data': routes,
            'load_data': loads
        }, fp, indent=4)
        
        print('results written to ', out_path)
