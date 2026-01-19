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
from simnibs_memoslap_utils.utils import get_focality_for_fixed_threshold


# get focalities of E-fields for fixed threshold values
#
# Focality is defined as area outside the ROI that exceeds
# the given threshold values

# path to a results folder       
results_folder = '/home/axelt/INN/axel/nobackup/travis/simulations/B1/P1_target_0900'
# list of one or more thresholds of interest
thresholds = [0.2, 0.3] 

res = get_focality_for_fixed_threshold(results_folder, thresholds)

print('results for '+results_folder)
print('results for E_magn:')
print('radii:')
print(res['radii'])
print('focality in [mm^2] for fixed threshold of 0.2 V/m:')
print(res[0.2]['E_magn'])
print('focality in [mm^2] for fixed threshold of 0.3 V/m:')
print(res[0.3]['E_magn'])