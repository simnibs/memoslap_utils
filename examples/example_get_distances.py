# -*- coding: utf-8 -*-
"""
Created on Mon Sept 04 2023

@author: axthi
"""
import sys

# --------------------------------------------------
# NOTE: Update this path to fit to your installation
path_to_utils = 'C:/Users/axthi/simnibsN/memoslap/'
sys.path.append(path_to_utils)
from simnibs_memoslap_utils.utils import get_scalp_skull_csf_thickness_for_subject


# get first quartiles, median and third quartiles of the scalp and skull 
# thicknesses in cylinders underneath the center electrode positions
subject_path = '../../m2m_ernie3/' # m2m-folder path

[Q1_scalp, med_scalp, Q3_scalp,
 Q1_bone, med_bone, Q3_bone,
 Q1_csf, med_csf, Q3_csf] = get_scalp_skull_csf_thickness_for_subject(subject_path,
                                                                     add_cerebellum=True,
                                                                     radius=20.)

print('scalp, first quartiles:')
print(Q1_scalp)
print('scalp, medians:')
print(med_scalp)
print('scalp, third quartiles:')
print(Q3_scalp)
print('skull, first quartiles:')
print(Q1_bone)
print('skull, medians:')
print(med_bone)
print('skull, third quartiles:')
print(Q3_bone)
print('csf, first quartiles:')
print(Q1_csf)
print('csf, medians:')
print(med_csf)
print('csf, third quartiles:')
print(Q3_csf)









