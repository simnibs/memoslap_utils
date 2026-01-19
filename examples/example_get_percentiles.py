# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 19:03:57 2025

@author: axthi
"""
import sys

# --------------------------------------------------
# NOTE: Update this path to fit to your installation
path_to_utils = '/mrhome/axelt/memoslap_utils/memoslap'
sys.path.append(path_to_utils)
from simnibs_memoslap_utils.utils import get_percentiles


# get percentiles of E-fields in the ROIs

# path to a results folder       
results_folder = '/home/axelt/INN/axel/nobackup/travis/simulations/B1/P1_target_0900'
# list of one or more percentiles of interest
percentiles = [50, 95] 

res = get_percentiles(results_folder, percentiles)

print('results for '+results_folder)
print('results for E_magn:')
print('radii:')
print(res['radii'])
print('50 percentile:')
print(res[50]['E_magn'])
print('95 percentile:')
print(res[95]['E_magn'])
