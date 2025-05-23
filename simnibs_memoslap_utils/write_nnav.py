# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 21:58:30 2023

@author: axthi
"""

import numpy as np
import os

from simnibs import mesh_io, sim_struct, __version__
from simnibs.utils.file_finder import SubjectFiles
from simnibs import localite, brainsight

version = [int(x) for x in __version__.split('.')[:3]]
isSimNIBS4 = version[0]>3
isSimNIBS4xx = (version[1]>0 or version[2]>0) and isSimNIBS4
isSimNIBS402 = (version[1]>0 or (version[1]==0 and version[2]>1)) and isSimNIBS4

def _make_matsimnibs(center, nodes, normals):
    # select z orthogonal to skin
    closest = np.argmin(np.linalg.norm(nodes - center, axis=1))
    z = -normals[closest]
    # otherwise make just a random orthonormal system
    y = np.zeros_like(z)
    y[np.argmin(np.abs(z))] = 1.
    # Orthogonalize y
    y -= z * y.dot(z)
    y /= np.linalg.norm(y)
    # Determine x
    x = np.cross(y, z)
    # matsimnibs
    matsimnibs = np.zeros((4, 4), dtype=float)
    matsimnibs[:3, 0] = x
    matsimnibs[:3, 1] = y
    matsimnibs[:3, 2] = z
    matsimnibs[:3, 3] = center
    matsimnibs[3, 3] = 1
    return matsimnibs 



def write_nnav_files(subject_path, res_list, pos_center, pos_surround):
    """
    write out electrode positions for import in localite or brainsight

    Parameters
    ----------
    subject_path : string
        m2m-folder.
    res_list : dict
        pathnames to the result meshes (on individual GM surfaces) for each radius
    pos_center : np.array
        position of center electrode.
    pos_surround : dict
        dictionary with the surround electrode positions for each radius
        
    Returns
    -------
    None.

    """
    label_skin = 1005
    subject_files = SubjectFiles(subpath=subject_path)
    
    # get skin nodes and normals
    m = mesh_io.read_msh(subject_files.fnamehead)
    m = m.crop_mesh(elm_type = 2)
    m = m.crop_mesh(tags = label_skin)
    skin_nodes = m.nodes.node_coord
    skin_normals = m.nodes_normals().value
    
    # create matsimnibs for each electrode position and save
    for radius, fname_msh in res_list.items():
        tmslist = sim_struct.TMSLIST()
        p = tmslist.add_position()
        p.matsimnibs = _make_matsimnibs(pos_center, skin_nodes, skin_normals)
        names = ['center_electrode']
        
        for i in range(pos_surround[radius].shape[0]):
            p = tmslist.add_position()
            p.matsimnibs = _make_matsimnibs(pos_surround[radius][i], skin_nodes, skin_normals)
            names.append('surround_electrode_' + str(i+1))
    
        fn_out = os.path.splitext(fname_msh)[0]
    
        localite().write(tmslist, fn_out+'localite_RAS',
                         names=names, overwrite=True,
                         out_coord_space="RAS")
        localite().write(tmslist, fn_out+'localite_LPS',
                         names=names, overwrite=True,
                         out_coord_space="LPS")
        
        if isSimNIBS4xx:
            brainsight().write(tmslist, fn_out+'brainsight_RAS',
                               names=names, overwrite=True)
        else:
            brainsight().write(tmslist, fn_out+'brainsight_LPS',
                               names=names, overwrite=True,
                               out_coord_space="World")
            brainsight().write(tmslist, fn_out+'brainsight_RAS',
                               names=names, overwrite=True,
                               out_coord_space="NifTI:Scanner")
    