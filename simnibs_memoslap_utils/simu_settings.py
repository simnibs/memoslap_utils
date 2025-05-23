# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 21:08:42 2023

@author: axthi
"""


from simnibs import sim_struct

def get_simu_template():
    """
    returns standard simulation settings
 
    Returns
    -------
    S : SESSION sim-struct
        a SESSION sim-struct with standard simulation settings defined,
        including the current flow through the center channel and the properties
        of the center electrode
    EL_surround : ELECTRODE sim-struct
        an ELECTRODE sim-struct with the properties of the surround electrodes

    """
    S = sim_struct.SESSION()
    S.map_to_surf = False
    S.map_to_fsavg = False
    S.open_in_gmsh = False
    
    tdcs_list = S.add_tdcslist()
    
    # properties of centre electrode
    EL_center = tdcs_list.add_electrode()
    EL_center.shape = 'ellipse'  # round shape
    EL_center.dimensions = [20, 20]  # 20 mm diameter
    EL_center.thickness = [2, 1]  # 2 mm rubber electrodes on top of 1 mm gel layer
    
    # properties of surround electrodes
    EL_surround = sim_struct.ELECTRODE()
    EL_surround.shape = 'ellipse'  
    EL_surround.dimensions = [20, 20]
    EL_surround.thickness = [2, 1]

    return S, EL_surround