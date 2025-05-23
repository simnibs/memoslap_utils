# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 19:03:57 2025

@author: axthi
"""
import sys

# --------------------------------------------------
# NOTE: Update this path to fit to your installation
path_to_utils = 'C:/Users/axthi/simnibs/memoslap_utils/'
sys.path.append(path_to_utils)
from simnibs_memoslap_utils.utils import get_get_tissue_volumes_for_subject


# get tissue volumes in cylinders underneath the center positions
subject_path = 'm2m_sub01'

(vol_WM, vol_GM, vol_CSF, vol_BONE, vol_SCALP) = get_get_tissue_volumes_for_subject(subject_path)


print('WM volume (in mm3):')
print(vol_WM)
print('GM volume (in mm3):')
print(vol_GM)
print('CSF volume (in mm3):')
print(vol_CSF)
print('BONE volume (in mm3):')
print(vol_BONE)
print('SCALP volume (in mm3):')
print(vol_SCALP)

