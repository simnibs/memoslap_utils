# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 19:03:57 2025

@author: axthi
"""
import sys
import numpy as np

# --------------------------------------------------
# NOTE: Update this path to fit to your installation
path_to_utils = '/mrhome/axelt/memoslap_utils/memoslap'
sys.path.append(path_to_utils)
import simnibs_memoslap_utils as smu


# general settings
project_nr = 2
exp_condition = 'target' # 'target' or 'control'
subject_path = 'm2m_4150' # m2m-folder path
results_basepath = 'tests' # results will be placed in subfolder of results_basepath

shift_distance = 30.0 # shift the center by up to 30 mm left-right and up-down
shift_steps = 7.5 # 7.5 mm spacing between neighboring center positions

# load project settings
project = smu.projects[project_nr][exp_condition]

# run simulations
res = smu.run_line(subject_path, project, results_basepath, shift_distance, shift_steps)

# some results (are also saved in simnibs_memoslap_results.pkl in the results directory)
for i in range(2):
    print(' ')
    print('line '+str(i+1))
    print('shifts of center electrode (in [mm]):')
    print(str(res[i]['pos_shifts']))
    print('median E_magn in ROI (in [V/m]):')
    print(str(res[i]['roi_median']['E_magn']))
    print('focality (in [mm^2]):')
    print(str(res[i]['focality']['E_magn']))
    print('product of median in ROI with inverse focality')
    print(str(1./np.array(res[i]['focality']['E_magn'])*np.array(res[i]['roi_median']['E_magn'])))
    