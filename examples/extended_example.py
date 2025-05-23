# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 16:06:50 2023

@author: axthi
"""

# This example performs the main steps that are normally done inside smu.run
# Hopefully useful for debugging and development...

import sys
# NOTE: update this path to fit to your installation
path_to_utils = 'C:/Users/axthi/simnibsN/memoslap/'
sys.path.append(path_to_utils)

from simnibs import __version__
isSimNIBS4 = int(__version__[0])>3
import simnibs_memoslap_utils as smu 

# general settings
project_nr = 4 # memoslap project nr
exp_condition = 'target' # 'target' or 'control'
subject_path = '../../m2m_ernie3/' # m2m-folder path
results_basepath = '../../tests' # results will be added as subfolder of results_basepath
add_cerebellum = True
map_to_fsavg = True

# load project settings
project = smu.projects[project_nr][exp_condition]

# update settings (optional)
project.condition = 'optimal' # select other method to get center electrode position
project.current = 0.001 # change current of center electrode
project.radius = [50.0, 60.0, 70.0] # test a range of radii

# create a coarse cerebellum central gm surface and
# add it to the m2m-folder content (only for charm results)
if add_cerebellum:
    smu.create_cereb_surface(subject_path)

# load middle gm surfaces and add the mask as node data
m_surf = smu.get_central_gm_with_mask(subject_path,
                                  project.hemi, 
                                  project.fname_roi,
                                  project.mask_type,
                                  add_cerebellum
                                  )

# get position of center electrode
pos_center = smu.get_center_pos(m_surf, subject_path, project.condition, project.el_name)

# perform the following steps:
#   * set up the FEM simulation (sets also the surround electrode positions)
#   * run the FEM simulations
#   * map results onto the middle GM surfaces
#   * map results of lh and rh to fsaverage (optional)                                  
pos_surround, results_path, res_list, res_list_raw = smu.run_FEMs(subject_path,
                                                                  project,
                                                                  results_basepath,
                                                                  pos_center,
                                                                  m_surf,
                                                                  map_to_fsavg
                                                                  )

# get field medians and focality
res_summary = smu.analyse_simus(res_list)
    
# export electrode positions for use with neuronavigation (only simnibs4)
if isSimNIBS4:
    smu.write_nnav_files(subject_path, res_list, pos_center, pos_surround)
       