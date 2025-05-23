# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 11:25:41 2023

@author: axthi
"""
import sys

# --------------------------------------------------
# NOTE: Update this path to fit to your installation
path_to_utils = 'C:/Users/axthi/simnibs/memoslap/'
sys.path.append(path_to_utils)
from simnibs_memoslap_utils.utils import write_roi_annots


subject_path = 'm2m_ernie222' # m2m-folder path
results_basepath = 'annot_files' # results will be placed in subfolder of results_basepath


write_roi_annots(subject_path, results_basepath, add_cerebellum=True, write_nii=True)
