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
from simnibs_memoslap_utils.utils import get_areas_for_subject


subject_path = 'm2m_ernie3' # m2m-folder path
areas = get_areas_for_subject(subject_path)
print(areas)
