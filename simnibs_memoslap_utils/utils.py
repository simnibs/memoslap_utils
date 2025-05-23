# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:11:37 2023

@author: axthi
"""
import numpy as np
import nibabel as nib
import gc
import os
from copy import deepcopy
from scipy.spatial.distance import cdist
from simnibs import mesh_io, file_finder

from .preparation import create_cereb_surface, get_central_gm_with_mask, get_center_pos
from .write_nnav import _make_matsimnibs
from .project_settings import projects

# NOTE: Replace this by import of SimNIBS ElementTags once available
from .mesh_element_properties import ElementTags


def _get_surface_cylinder(m, label, matrix, radius=20):
    ''' extracts surface cylinder underneath electrode '''
    # extract surface
    m_surf = m.crop_mesh(tags = label)
    
    # tranform into electrode coordinate space
    node_pos = m_surf.nodes.node_coord
    node_pos = np.matmul(np.linalg.inv(matrix),
                         np.hstack((node_pos, np.ones((node_pos.shape[0],1)))).T
                         )
    m_surf.nodes.node_coord = node_pos[:3,:].T
        
    # crop cylinder
    idx = np.argwhere(np.linalg.norm(m_surf.nodes.node_coord[:,:2],
                                     axis=1)<=radius
                      )+1
    m_surf = m_surf.crop_mesh(nodes=idx)
    
    # keep surface component closest to electrode
    components = m_surf.elm.connected_components() # returns 1-based elm indices 
    baricenters = m_surf.elements_baricenters().value
    meddist = [np.median(baricenters[c-1][:,2]) for c in components]
    m_surf = m_surf.crop_mesh(elements=components[np.argmin(meddist)])
    
    # transform back to original space
    node_pos = m_surf.nodes.node_coord
    node_pos = np.matmul(matrix,
                         np.hstack((node_pos, np.ones((node_pos.shape[0],1)))).T
                         )
    m_surf.nodes.node_coord = node_pos[:3,:].T
    return m_surf


def _return_min_distance_Q1medianQ3(coords1, coords2):
    dist_mat = cdist(coords1, coords2)
    min_per_column = dist_mat.min(axis=0)
    min_per_row = dist_mat.min(axis=1)
    all_mins = np.hstack((min_per_column, min_per_row))

    Q1 = np.percentile(all_mins, 25)
    med = np.median(all_mins)
    Q3 = np.percentile(all_mins, 75)
    
    return Q1, med, Q3


def _return_GM_distance_Q1medianQ3(m, AABBTree, points, directions):
    indices, intercpt_pos = m.intersect_ray(points, -directions, AABBTree)
    indices_pts = indices[:,0]
    dist = np.linalg.norm(points[indices_pts] - intercpt_pos, axis=1)
    idx_uniq = np.unique(indices_pts)    
    dist_uniq = np.array([np.min(dist[indices_pts == i]) for i in idx_uniq])
    
    Q1 = np.percentile(dist_uniq, 25)
    med = np.median(dist_uniq)
    Q3 = np.percentile(dist_uniq, 75)
    
    end_pts = points[idx_uniq] - directions[idx_uniq]*dist_uniq.reshape([-1,1]) # for debugging
    
    return Q1, med, Q3, dist_uniq, points[idx_uniq], end_pts
    
    

def get_matsimnibs(subject_path, pos_centers):
    """
    return the matsimnibs matrices for the given center positions

    Parameters
    ----------
    subject_path : string
        path to m2m-folder.
    pos_centers : np.array
        array of shape (N_positions, 3).

    Returns
    -------
    matsimnibs : np.array
        array of shape (N_positions, 4, 4).

    """
    # load mesh and get skin nodes and normals
    # (required for creating the normal direction under the electrode)
    ff = file_finder.SubjectFiles(subpath = subject_path)
    m=mesh_io.read_msh(ff.fnamehead)
    
    m = m.crop_mesh(elm_type = 2)
    m = m.crop_mesh(tags = ElementTags.SCALP_TH_SURFACE)
    skin_nodes = m.nodes.node_coord
    skin_normals = m.nodes_normals().value
    
    matsimnibs = np.zeros((pos_centers.shape[0],4,4))
    for i, pos in enumerate(pos_centers):
        matsimnibs[i] = _make_matsimnibs(pos, skin_nodes, skin_normals)
    return matsimnibs


def get_scalp_skull_csf_thickness(subject_path, matsimnibs, radius=20.):
    """
    return scalp, skull and csf thicknesses for the positions defined by the 
    matsimnibs

    Parameters
    ----------
    subject_path : string
        path to m2m-folder.
    matsimnibs : np.array
        array of shape (N_positions, 4, 4).
    radius: float, optional
        radius of the cylinders used to cut out the surfaces. The default is 20.
        
    Returns
    -------
    Q1_scalp, med_scalp, Q3_scalp, Q1_bone, med_bone, Q3_bone, 
    Q1_csf, med_csf, Q3_csf : np.arrays of shape (N_positions,)
        first quartiles, median and third quartiles of the scalp, skull and csf 
        thicknesses in the cylinders defined by the matsimnibs and the radius

    """
    DEBUG = False
    
    # load mesh, recreate surfaces to ensure surfaces of interest are closed
    ff = file_finder.SubjectFiles(subpath = subject_path)
    m = mesh_io.read_msh(ff.fnamehead)
    m = m.crop_mesh(elm_type=4)
    
    # get brain surface
    m_brain = m.crop_mesh(tags = [ElementTags.WM, ElementTags.GM])
    m_brain.elm.tag1[:] = ElementTags.GM
    m_brain.elm.tag2[:] = m_brain.elm.tag1
    m_brain.reconstruct_unique_surface()
    m_brain = m_brain.crop_mesh(elm_type=2)
    AABBTree = m_brain.get_AABBTree()
    
    # get inner and outer skull surfaces and scalp surface
    idx = (m.elm.tag1 == ElementTags.WM) + (m.elm.tag1 == ElementTags.GM) + (m.elm.tag1 == ElementTags.BLOOD)
    m.elm.tag1[idx] = ElementTags.CSF # everything inside skull becomes CSF
    
    idx = (m.elm.tag1 == ElementTags.COMPACT_BONE) + (m.elm.tag1 == ElementTags.SPONGY_BONE)
    m.elm.tag1[idx] = ElementTags.BONE # combine bone types
    
    idx = (m.elm.tag1 == ElementTags.EYE_BALLS) + (m.elm.tag1 == ElementTags.MUSCLE)
    m.elm.tag1[idx] = ElementTags.SCALP # everything outside skull becomes scalp
    
    m.elm.tag2[:] = m.elm.tag1
    m.reconstruct_unique_surface(add_outer_as = ElementTags.SCALP)
    m = m.crop_mesh(elm_type=2)
        
    if DEBUG:
        m_dbg = deepcopy(m)
    
    # loop over matsimnibs to get csf, bone, scalp thicknesses
    Q1_scalp = np.zeros(matsimnibs.shape[0])
    med_scalp = np.zeros(matsimnibs.shape[0])
    Q3_scalp = np.zeros(matsimnibs.shape[0])
    
    Q1_bone = np.zeros(matsimnibs.shape[0])
    med_bone = np.zeros(matsimnibs.shape[0])
    Q3_bone = np.zeros(matsimnibs.shape[0])
    
    Q1_csf = np.zeros(matsimnibs.shape[0])
    med_csf = np.zeros(matsimnibs.shape[0])
    Q3_csf = np.zeros(matsimnibs.shape[0])
    
    for i, mat in enumerate(matsimnibs):
        # cut out surface cylinders underneath the electrode
        m_surf_SCALP = _get_surface_cylinder(m, ElementTags.SCALP_TH_SURFACE, mat, radius=radius)
        m_surf_BONE  = _get_surface_cylinder(m, ElementTags.BONE_TH_SURFACE, mat, radius=radius)
        m_surf_CSF   = _get_surface_cylinder(m, ElementTags.CSF_TH_SURFACE, mat, radius=radius)
        
        # measure distances between surface cylinders
        r = _return_min_distance_Q1medianQ3(m_surf_BONE.nodes.node_coord, 
                                            m_surf_SCALP.nodes.node_coord)
        Q1_scalp[i] = r[0]
        med_scalp[i] = r[1]
        Q3_scalp[i] = r[2]
        
        r = _return_min_distance_Q1medianQ3(m_surf_BONE.nodes.node_coord, 
                                            m_surf_CSF.nodes.node_coord)      
        Q1_bone[i] = r[0]
        med_bone[i] = r[1]
        Q3_bone[i] = r[2]
        
        r = _return_GM_distance_Q1medianQ3(m_brain, AABBTree,
                                           m_surf_CSF.nodes.node_coord,
                                           m_surf_CSF.nodes_normals().value)
        Q1_csf[i] = r[0]
        med_csf[i] = r[1]
        Q3_csf[i] = r[2]
        
        if DEBUG:
            mesh_io.write_geo_lines(r[4], r[5], 'debug.geo', values=np.vstack((r[3],r[3])).T,
                                    name=str(i), mode='ba')
        if DEBUG:
            for k, m_hlp in enumerate([m_surf_SCALP,m_surf_BONE,m_surf_CSF]):
                m_hlp.elm.tag1[:] = 101+i*10+k
                m_hlp.elm.tag2 = m_hlp.elm.tag1
                m_dbg = m_dbg.join_mesh(m_hlp)
                
    if DEBUG:
        m_dbg = m_dbg.join_mesh(m_brain)
        mesh_io.write_msh(m_dbg,'debug.msh')
    
    return Q1_scalp, med_scalp, Q3_scalp, Q1_bone, med_bone, Q3_bone, Q1_csf, med_csf, Q3_csf
    
    
def get_pos_centers_for_subject(subject_path, add_cerebellum=True):
    """
    returns the center positions for all projects (target and control)
    for a subject

    Parameters
    ----------
    subject_path : string
        path to m2m-folder.
    add_cerebellum : boolean, optional
        whether to add cerebellum to the surface (needed for P6). 
        The default is True.

    Returns
    -------
    pos_centers : dict of np.array
        The dict contains two arrays of length eight with the center positions
        of the projects, one for target and one for control.

    """
    pos_centers = dict()
    
    if add_cerebellum:
        create_cereb_surface(subject_path, add_cerebellum)
    
    for exp_condition in ['target', 'control']:
        pos_hlp = np.zeros((len(projects),3))
        
        for project_nr in range(1,len(projects)+1):
            project = projects[project_nr][exp_condition]
                        
            # get position of center electrode
            m_surf = get_central_gm_with_mask(subject_path,
                                              project.hemi,
                                              project.fname_roi,
                                              project.mask_type,
                                              add_cerebellum
                                                  )
            pos_hlp[project_nr-1] = get_center_pos(m_surf,
                                                   subject_path,
                                                   project.condition,
                                                   project.el_name
                                                   )   
        pos_centers.update({exp_condition: pos_hlp})
        
    return pos_centers


def get_scalp_skull_csf_thickness_for_subject(subject_path, add_cerebellum=True, radius=20.):
    """
    return scalp, skull and csf thicknesses for the target and control positions

    Parameters
    ----------
    subject_path : string
        path to m2m-folder.
    add_cerebellum : boolean, optional
        whether to add cerebellum to the surface (needed for P6). 
        The default is True.
    radius : float, optional
        radius of the cylinders used to cut out the surfaces. 

    Returns
    -------
    Q1_scalp, med_scalp, Q3_scalp, Q1_bone, med_bone, Q3_bone, 
    Q1_csf, med_csf, Q3_csf : dict
        dict containing first quartiles, median and third quartiles of the scalp
        and skull thicknesses in the cylinders defined by the radius, 
        separately for the target and control positions

    """
    # get center positions for all projects
    print('getting center positions')
    pos_centers = get_pos_centers_for_subject(subject_path, add_cerebellum)
    pos_centers_stacked = np.vstack((pos_centers['target'], 
                                     pos_centers['control']))
    
    # get matsimnibs for the center positions
    print('getting matsimnibs for the center positions')
    matsimnibs = get_matsimnibs(subject_path, pos_centers_stacked)
    
    # extract surfaces in cylinders below center positions and calculate thickness
    print('getting scalp and skull thickness underneath center positions')
    [Q1_sc, med_sc, Q3_sc,
     Q1_bo, med_bo, Q3_bo,
     Q1_cs, med_cs, Q3_cs] = get_scalp_skull_csf_thickness(subject_path, 
                                                           matsimnibs, 
                                                           radius=radius
                                                           )                                                
    len_target  = pos_centers['target'].shape[0]
    Q1_scalp  = {'target' : Q1_sc[:len_target],  'control' : Q1_sc[len_target:]}
    med_scalp = {'target' : med_sc[:len_target], 'control' : med_sc[len_target:]}
    Q3_scalp  = {'target' : Q3_sc[:len_target],  'control' : Q3_sc[len_target:]}
    
    Q1_bone   = {'target' : Q1_bo[:len_target],  'control' : Q1_bo[len_target:]}
    med_bone  = {'target' : med_bo[:len_target], 'control' : med_bo[len_target:]}
    Q3_bone   = {'target' : Q3_bo[:len_target],  'control' : Q3_bo[len_target:]}
    
    Q1_csf   = {'target' : Q1_cs[:len_target],  'control' : Q1_cs[len_target:]}
    med_csf  = {'target' : med_cs[:len_target], 'control' : med_cs[len_target:]}
    Q3_csf   = {'target' : Q3_cs[:len_target],  'control' : Q3_cs[len_target:]}
    
    return Q1_scalp, med_scalp, Q3_scalp, Q1_bone, med_bone, Q3_bone, Q1_csf, med_csf, Q3_csf


def get_areas_for_subject(subject_path, add_cerebellum=True):
    """
    calculates the roi areas for all projects (target and control)
    for a subject

    Parameters
    ----------
    subject_path : string
        path to m2m-folder.
    add_cerebellum : boolean, optional
        whether to add cerebellum to the surface (needed for P6). 
        The default is True.

    Returns
    -------
    roi_areas : dict of np.array
        The dict contains two arrays of length eight with the roi
        areas of the projects, one for target and one for control.

    """
    roi_areas = dict()
    
    if add_cerebellum:
        create_cereb_surface(subject_path, add_cerebellum)
    
    for exp_condition in ['target', 'control']:
        areas_hlp = np.zeros(len(projects))
        
        for project_nr in range(1,len(projects)+1):
            project = projects[project_nr][exp_condition]
            
            # get surface with mask as nodedata field
            m_surf = get_central_gm_with_mask(subject_path,
                                              project.hemi,
                                              project.fname_roi,
                                              project.mask_type,
                                              add_cerebellum
                                              )
            mask_idx = m_surf.field['mask'].value
            
            # get node areas
            areas = m_surf.nodes_areas().value
            assert len(areas) == len(mask_idx)
            
            # calculate area of mask
            areas_hlp[project_nr-1] = np.sum(areas[mask_idx])
            
            print('P'+str(project_nr)+' '
                  +exp_condition+': '+
                  "{:.1f}".format(areas_hlp[project_nr-1]))
        roi_areas.update({exp_condition: areas_hlp})
        
    return roi_areas


def get_tissue_volumes(subject_path, matsimnibs, radius=20., max_depth = 40.):
    """
    returns tissue volumes in cylinders underneath the positions

    Parameters
    ----------
    subject_path : string
        path to m2m-folder.
    matsimnibs : np.array
        array of shape (N_positions, 4, 4).
    radius: float, optional
        radius of the cylinders. The default is 20.
    max_depth: float, optional
        maximal depth of the cylinders. The default is 40.
        
    Returns
    -------
    vol_WM, vol_GM, vol_CSF, vol_BONE, vol_SCALP: np.arrays of shape (N_positions,)
        tissue volumes in mm3 inside the cylinders 
        defined by the matsimnibs, the radius and maximal depth

    """
    DEBUG = False
    
    # load mesh, recreate surfaces to ensure surfaces of interest are closed
    ff = file_finder.SubjectFiles(subpath = subject_path)
    m = mesh_io.read_msh(ff.fnamehead)
    m = m.crop_mesh(elm_type=4)
    
    # relabel BLOOD to CSF
    idx = m.elm.tag1 == ElementTags.BLOOD
    m.elm.tag1[idx] = ElementTags.CSF # everything inside skull becomes CSF
    
    # combine COMPACT_BONE and SPONGY_BONE
    idx = (m.elm.tag1 == ElementTags.COMPACT_BONE) + (m.elm.tag1 == ElementTags.SPONGY_BONE)
    m.elm.tag1[idx] = ElementTags.BONE # combine bone types
    
    # relabel EYE_BALLS and MUSCLE to SCALP
    idx = (m.elm.tag1 == ElementTags.EYE_BALLS) + (m.elm.tag1 == ElementTags.MUSCLE)
    m.elm.tag1[idx] = ElementTags.SCALP # everything outside skull becomes scalp
    
    # get tet volumes
    ed = m.elements_volumes_and_areas()
    m.add_element_field(ed,'tet_volumes')
    
    
    vol_GM = np.zeros(matsimnibs.shape[0])
    vol_WM = np.zeros(matsimnibs.shape[0])
    vol_CSF = np.zeros(matsimnibs.shape[0])
    vol_BONE = np.zeros(matsimnibs.shape[0])
    vol_SCALP = np.zeros(matsimnibs.shape[0])
    for i, matrix in enumerate(matsimnibs):
    
        # tranform into electrode coordinate space
        node_pos = deepcopy(m.nodes.node_coord)
        node_pos = np.matmul(np.linalg.inv(matrix),
                             np.hstack((node_pos, np.ones((node_pos.shape[0],1)))).T
                             )
        node_pos = node_pos[:3,:].T
            
        # crop cylinder
        idx = np.linalg.norm(node_pos[:,:2],axis=1)<=radius
        idx *= node_pos[:,2] <= max_depth
        m_hlp = m.crop_mesh(nodes=np.argwhere(idx)+1)
        
        # get WM, GM, CSF, BONE and SCALP volumes
        idx = m_hlp.elm.tag1 == ElementTags.WM
        vol_WM[i] = np.sum(m_hlp.field['tet_volumes'].value[idx])
        
        idx = m_hlp.elm.tag1 == ElementTags.GM
        vol_GM[i] = np.sum(m_hlp.field['tet_volumes'].value[idx])
        
        idx = m_hlp.elm.tag1 == ElementTags.CSF
        vol_CSF[i] = np.sum(m_hlp.field['tet_volumes'].value[idx])
        
        idx = m_hlp.elm.tag1 == ElementTags.BONE
        vol_BONE[i] = np.sum(m_hlp.field['tet_volumes'].value[idx])
        
        idx = m_hlp.elm.tag1 == ElementTags.SCALP
        vol_SCALP[i] = np.sum(m_hlp.field['tet_volumes'].value[idx])
        
        if DEBUG:
            m_hlp.elm.tag2 = m_hlp.elm.tag1
            mesh_io.write_msh(m_hlp,'debug'+str(i)+'.msh')
            
    return vol_WM, vol_GM, vol_CSF, vol_BONE, vol_SCALP


def get_get_tissue_volumes_for_subject(subject_path, add_cerebellum=True, radius=20., max_depth = 40.):
    """
    return tissue volumes in cylinders under the center electrode
    for the target and control positions

    Parameters
    ----------
    subject_path : string
        path to m2m-folder.
    add_cerebellum : boolean, optional
        whether to add cerebellum to the surface (needed for P6). 
        The default is True.
    radius : float, optional
        radius of the cylinders. The default is 20.
    max_depth : float, optional
        maximal depth of the cylinders. The default is 40.
        
    Returns
    -------
    vol_WM, vol_GM, vol_CSF, vol_BONE, vol_SCALP : dict
        dict containing the tissue volumes in mm3 inside the cylinders 
        under the center electrodes, separately for the target and 
        control positions

    """
    # get center positions for all projects
    print('getting center positions')
    pos_centers = get_pos_centers_for_subject(subject_path, add_cerebellum)
    pos_centers_stacked = np.vstack((pos_centers['target'], 
                                     pos_centers['control']))
    
    # get matsimnibs for the center positions
    print('getting matsimnibs for the center positions')
    matsimnibs = get_matsimnibs(subject_path, pos_centers_stacked)
    
    # loop over projects
    (v_WM, v_GM, v_CSF, v_BONE, v_SCALP) = get_tissue_volumes(subject_path,
                                                              matsimnibs,
                                                              radius,
                                                              max_depth)
    
    len_target  = pos_centers['target'].shape[0]
    vol_WM = {'target' : v_WM[:len_target], 'control' : v_WM[len_target:]}
    vol_GM = {'target' : v_GM[:len_target], 'control' : v_GM[len_target:]}
    vol_CSF = {'target' : v_CSF[:len_target], 'control' : v_CSF[len_target:]}
    vol_BONE = {'target' : v_BONE[:len_target], 'control' : v_BONE[len_target:]}
    vol_SCALP = {'target' : v_SCALP[:len_target], 'control' : v_SCALP[len_target:]}
    
    return vol_WM, vol_GM, vol_CSF, vol_BONE, vol_SCALP


def write_roi_annots(subject_path, results_basepath='.', add_cerebellum=True, write_nii=True):
    """
    writes the project ROIs as annot files for further use with FreeSurfer

    NOTE: Cerebellar ROIs starting with "cereb." can be overlaid on 
          m2m_<subID>\surfaces\cerebellum.central.gii
          However, as this not a standard surface, FreeSurfer tools might not work

    Parameters
    ----------
    subject_path : string
        path to m2m-folder.
    results_basepath : string, optional
        folder to which the results will be added as subfolder. The default is '.'
    add_cerebellum : boolean, optional
        whether to add cerebellum to the surface (needed for P6). 
        The default is True.
    write_nii : boolean, optional
        whether to additionally write the ROIs as voxel mask in subject space
        The default is True.
        
    """
    if add_cerebellum:
        create_cereb_surface(subject_path, add_cerebellum)
    
    # create results directory
    if not os.path.isdir(results_basepath):
        try:
            os.mkdir(results_basepath)
        except:
            raise IOError('Could not create directory '+results_basepath)
    
    subject_files = file_finder.SubjectFiles(subpath=subject_path)
    results_path = os.path.abspath(os.path.join(results_basepath, 
                                                'roi_annots_'+subject_files.subid
                                                )
                                  )
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    
    # loop over projects
    hemi_tag_map = {'lh': 1001, 'rh': 1002, 'cereb': 1003}
    for exp_condition in ['target', 'control']:
        for project_nr in range(1,len(projects)+1):
            project = projects[project_nr][exp_condition]
            
            # get surface with mask as nodedata field
            m_surf = get_central_gm_with_mask(subject_path,
                                              project.hemi,
                                              project.fname_roi,
                                              project.mask_type,
                                              add_cerebellum
                                              )
            
            # write mask as annot
            m_hlp = m_surf.crop_mesh(tags = hemi_tag_map[project.hemi])
            mask_idx = m_hlp.field['mask'].value
                    
            labels =-1*np.ones(m_hlp.nodes.nr,dtype='int')
            labels[mask_idx] = 0
            fn_out = os.path.join(results_path,
                                  project.hemi+'.P'+str(project_nr)+'_'+exp_condition+'.annot')
            
            nib.freesurfer.io.write_annot(fn_out, 
                                          labels,
                                          ctab = np.asarray([255, 0, 0, 200, 200]).reshape((1,5)),
                                          names = ['roi'],
                                          fill_ctab=True)

            
            if write_nii:
                fn_out = os.path.join(results_path,
                                               project.hemi+'.P'+str(project_nr)+'_'+exp_condition+'.nii.gz')
                
                img = nib.load(subject_files.reference_volume)
                affine = img.affine
                iM = np.linalg.inv(affine)
                
                pos = m_hlp.nodes.node_coord[mask_idx]
                vox_idx = np.round(iM[:3, :3].dot(pos.T) + iM[:3, 3, None]).astype(int)
        
                idx_keep =  (vox_idx[0] >= 0) * (vox_idx[0] < img.shape[0])
                idx_keep *= (vox_idx[1] >= 0) * (vox_idx[1] < img.shape[1])
                idx_keep *= (vox_idx[2] >= 0) * (vox_idx[2] < img.shape[2])
                vox_idx = vox_idx[:,idx_keep]

                img = np.squeeze(np.zeros(img.shape, dtype = np.uint8))
                img[vox_idx[0],vox_idx[1],vox_idx[2]] = 1
                img = nib.Nifti1Image(img, affine)
                nib.save(img,fn_out)
                
                del img, vox_idx, pos

            del m_hlp, mask_idx, labels
            gc.collect()
            