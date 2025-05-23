# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 11:12:54 2023

@author: axthi
"""

import sys
# --------------------------------------------------
# NOTE: Update this path to fit to your installation
path_to_utils = 'C:/Users/axthi/simnibsN/memoslap/'
sys.path.append(path_to_utils)

import simnibs_memoslap_utils as smu 

# general settings
project_nr = 1
exp_condition = 'target' # 'target' or 'control'
subject_path = '../../m2m_ernie3/' # m2m-folder path
results_basepath = '../../tests' # results will be placed in subfolder of results_basepath

# load project settings
project = smu.projects[project_nr][exp_condition]

# run calculations
smu.run(subject_path, project, results_basepath,
        add_cerebellum=True, map_to_fsavg=True)


# ---------------------------------------------
# NOTE: Overwrite project settings to customize
#
# Example (uncomment to run):
#
# # load project settings
# project = smu.projects[project_nr][exp_condition]

# # update settings
# project.condition = 'closest' # select other method to get center electrode position
# project.current = 0.001 # change current of center electrode
# project.radius = [50.0, 60.0, 70.0] # loop over several radii

# # run
# [res_list, 
#  res_list_raw,
#  pos_center,
#  pos_surround,
#  res_summary] = smu.run(subject_path, project, results_basepath,
#                         add_cerebellum=True, map_to_fsavg=True)
