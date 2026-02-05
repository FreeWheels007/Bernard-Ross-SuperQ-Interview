# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 09:21:32 2026

@author: Bernard

simple script to read json inputs into python numpy objects 
"""

import json
import numpy as np

def read_data(file_path):
    with open(file_path, 'r') as fp:
        input_data = json.load(fp)
        
    distance_matrix = np.array(input_data['distance_matrix'])
    demand_vector = np.array(input_data['demands'])

    number_vehicles = input_data['vehicles']['count']
    number_locations = input_data['nodes']['total']
    vehicle_capacity = input_data['vehicles']['capacity_per_vehicle']
    
    return distance_matrix, demand_vector, number_locations, number_vehicles, vehicle_capacity