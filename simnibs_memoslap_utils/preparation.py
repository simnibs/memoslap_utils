# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 16:50:29 2021

@author: axthi
"""

import gc
import copy
import nibabel as nib
import numpy as np
import os
from scipy import ndimage
from scipy.ndimage.filters import uniform_filter
import scipy.ndimage.morphology as mrph

from simnibs import mesh_io, sim_struct, __version__
from simnibs.simulation import fem
from simnibs.utils.file_finder import SubjectFiles
from simnibs.utils.simnibs_logger import logger

version = [int(x) for x in __version__.split('.')[:3]]
isSimNIBS4 = version[0]>3
isSimNIBS4xx = (version[1]>0 or version[2]>0) and isSimNIBS4
isSimNIBS402 = (version[1]>0 or (version[1]==0 and version[2]>1)) and isSimNIBS4
if isSimNIBS4:
    from simnibs.utils import transformations
    from simnibs.utils.file_finder import get_reference_surf
    from simnibs.segmentation.charm_utils import _get_largest_components
    from simnibs.segmentation.marching_cube import marching_cube
else:
    from simnibs.msh import transformations
    from simnibs.utils.file_finder import templates

if isSimNIBS402:
    from simnibs.utils import cond_utils as cond
else:
    from simnibs.simulation import cond
    
def _convert_fsavg_mask(fn_mask_fsspace, hemi, subpath):
    ''' convert mask roi from fsaverage to individual space and get positions
    '''
    assert hemi in ['lh', 'rh']

    subject_files = SubjectFiles(subpath=subpath)
    if isSimNIBS4:
        fn_sphere = get_reference_surf(hemi, 'sphere')
        fn_reg = subject_files.get_surface(hemi, 'sphere_reg')
        fn_central = subject_files.get_surface(hemi, 'central')
    else:
        if hemi == 'lh':
            fn_sphere = templates.cat_lh_sphere_ref
            fn_reg = subject_files.lh_reg
            fn_central = subject_files.lh_midgm
        else:
            fn_sphere = templates.cat_rh_sphere_ref
            fn_reg = subject_files.rh_reg
            fn_central = subject_files.rh_midgm

    surf_sphere = mesh_io.read_gifti_surface(fn_sphere)
    try:
        idx = nib.freesurfer.io.read_label(fn_mask_fsspace)
        if len(idx) == 0:
            raise ValueError()
        idx_mask = np.zeros(surf_sphere.nodes.nr, dtype=np.float32)
        idx_mask[idx] = 1.
    except:
        idx_mask = nib.freesurfer.io.read_morph_data(fn_mask_fsspace)

    if isSimNIBS4xx:
        morph = transformations.SurfaceMorph(surf_sphere, 
                                             mesh_io.read_gifti_surface(fn_reg), 
                                             method="nearest")
        idx_mask = morph.transform(idx_mask) > 0.0001
    else:
        idx_mask, _ = transformations._surf2surf(
                    idx_mask,
                    surf_sphere,
                    mesh_io.read_gifti_surface(fn_reg)
                    )
    idx_mask = idx_mask > 0.0001
    gm_surf =  mesh_io.read_gifti_surface(fn_central)
    central_pos = gm_surf.nodes[idx_mask]

    return idx_mask, central_pos


def _map_roi(subj_files, fname_roi):

    target_im = nib.load(subj_files.T1_upsampled)
    target_dim = list(target_im.get_fdata().shape)
    
    im_deformation = nib.load(subj_files.conf2mni_nonl)

    im_roi = nib.load(fname_roi)
    transformed_data = transformations.volumetric_nonlinear(
                                (im_roi.get_fdata(), im_roi.affine), 
                                (im_deformation.get_fdata(), im_deformation.affine), 
                                target_space_affine = target_im.affine,
                                target_dimensions = target_dim, 
                                intorder=0
                                )
    transformed_data = np.squeeze(transformed_data)

    return transformed_data, target_im.affine


def _convert_MNImask(fn_mask, m, subpath):
    ''' maps a mask from MNI space to mesh nodes
    '''
    subj_files = SubjectFiles(subpath = subpath)     
    # convert to subject space
    roi_buffer, roi_affine = _map_roi(subj_files, fn_mask)
    roi_buffer = roi_buffer.astype(np.uint16)
    # map on nodes
    nd = mesh_io.NodeData.from_data_grid(m, roi_buffer, roi_affine)
    nd.value = nd.value > 0
    m.add_node_field(nd,'mask')
    mask_pos = m.nodes.node_coord[nd.value, :]
    return m, mask_pos


def get_central_gm_with_mask(subpath, hemi, fn_mask_fsspace,
                             mask_type='curv', add_cerebellum = False):
    """ load bihemispheric GM and add mask as node data
        simnibs4: also cerebellum GM can be added
    
    Parameters
    ----------
    subpath : string
        m2m-folder
    hemi : string
        Defines on which hemisphere the mask is ('lh', 'rh' or 'cereb').
    fn_mask: string
        Path to the mask file
    mask_type : string, optional
        Indicates the type of mask ('curv' for masks in fsaverage space, 
        'mnivol' for masks in MNI space). The default is 'curv'.
    add_cerebellum : bool, optional
        whether to add the cerebellum central gm surface to the mesh. 
        The default is False.

    Returns
    -------
    m_surf : simnibs.mesh_io.Msh
        Central gm surfaces (labels: lh 1001, rh 1002, cerebellum 1003).
        The mask is added as nodedata field (get via m_surf.field['mask']).

    """

    assert hemi in ['lh', 'rh', 'cereb']
    if hemi == 'cereb' and not isSimNIBS4:
        raise ValueError('hemi cannot be cereb when using SimNIBS3')
        
    subject_files = SubjectFiles(subpath=subpath)

    fn_cereb = None
    if isSimNIBS4:
        fn_lh_central = subject_files.get_surface('lh', 'central')
        fn_rh_central = subject_files.get_surface('rh', 'central')
        
        if add_cerebellum:
            fn_cereb = os.path.join(subject_files.surface_folder, 
                                    'cerebellum.central.gii')
            if not os.path.exists(fn_cereb):
                raise FileNotFoundError(fn_cereb)
    else:
        fn_lh_central = subject_files.lh_midgm
        fn_rh_central = subject_files.rh_midgm

    m_surf = mesh_io.read_gifti_surface(fn_lh_central)
    m_surf.elm.tag1 = 1001 * np.ones(m_surf.elm.nr, dtype=int)
    nr_nodes_lh = m_surf.nodes.nr

    m_tmp = mesh_io.read_gifti_surface(fn_rh_central)
    m_tmp.elm.tag1 = 1002 * np.ones(m_tmp.elm.nr, dtype=int)
    nr_nodes_rh = m_tmp.nodes.nr
    m_surf = m_surf.join_mesh(m_tmp)
    
    if add_cerebellum:
        m_tmp = mesh_io.read_gifti_surface(fn_cereb)
        m_tmp.elm.tag1 = 1003 * np.ones(m_tmp.elm.nr, dtype=int)
        nr_nodes_cereb = m_tmp.nodes.nr
        m_surf = m_surf.join_mesh(m_tmp)
    
    m_surf.elm.tag2[:] = m_surf.elm.tag1
    
    if mask_type == 'mnivol':
        m_surf, mask_pos = _convert_MNImask(fn_mask_fsspace, m_surf, subpath)

    elif mask_type == 'curv':
        # requires surface registration to fsaverage, thus only for lh and rh
        idx_mask, _ = _convert_fsavg_mask(fn_mask_fsspace, hemi, subpath)
        
        if hemi == 'lh':
            idx_lh = idx_mask
        else:
            idx_lh = np.zeros(nr_nodes_lh,dtype=bool)
            
        if hemi == 'rh':
            idx_rh = idx_mask
        else:
            idx_rh = np.zeros(nr_nodes_rh,dtype=bool)
            
        if add_cerebellum:
            idx_cereb = np.zeros(nr_nodes_cereb,dtype=bool)
        else:
            idx_cereb = []
                
        nd=mesh_io.NodeData( np.hstack((idx_lh, idx_rh, idx_cereb)) )
        m_surf.add_node_field(nd,'mask')
        
    elif mask_type == '':
        nd=mesh_io.NodeData( np.zeros(m_surf.nodes.nr, dtype=bool) )
        m_surf.add_node_field(nd,'mask')
        
    else:
        raise ValueError(f"unknown mask_type: {mask_type}")
    
    return m_surf


def _get_outer_skin_points(m, tol: float = 1e-3, label_skin = 1005):
        """Return indices of points estimated to be on the outer skin surface
        (i.e., not those inside nasal cavities, ear canals etc.). Outer points
        are identified by looking for points which do not intersect the mesh in
        the direction of its normal. This is not perfect but seems to do a
        reasonable job of identifying the relevant points. These may then be
        used for projecting electrodes onto the surface.

        PARAMETERS
        ----------
        tol : float
            Tolerance for avoiding self-intersections.
        label_skin : int
            skin label (standard: 1005)
            
        RETURNS
        -------
        indices : ndarray
            Indices of the outer skin points (0-based).
        """
        assert tol > 0

        skin_faces = m.elm[m.elm.tag1 == label_skin, :3]
        subset = np.unique(skin_faces-1)
        m = mesh_io.Msh(mesh_io.Nodes(m.nodes.node_coord), mesh_io.Elements(skin_faces))

        subset = subset if len(subset) < m.nodes.nr else slice(None)
        n = m.nodes_normals().value[subset]
        # Avoid self-intersections by moving each point slightly along the test
        # direction
        idx = np.unique(m.intersect_ray(m.nodes.node_coord[subset] + tol * n, n)[0][:, 0])
        if isinstance(subset, slice):
            return np.setdiff1d(np.arange(m.nodes.nr), idx, assume_unique=True)
        else:
            return np.setdiff1d(subset, subset[idx], assume_unique=True)


def _relabel_internal_air(m, label_skin = 1005, label_new = 1099, keep_largest=True):
    ''' relabels skin in internal air cavities to something else;
        relevant for charm meshes
    '''
    m = copy.copy(m)
    # outer skin nodes
    idx_skinNodes = _get_outer_skin_points(m, label_skin = label_skin) + 1
    
    # internal air triangles
    idx_innerAirTri = (m.elm.elm_type == 2) * (m.elm.tag1 == label_skin)
    idx_innerAirTri *= ~np.any(np.in1d(m.elm.node_number_list, idx_skinNodes).reshape(-1, 4), axis=1)
    
    m.elm.tag1[idx_innerAirTri] = label_new
    
    if keep_largest:
        c=m.elm.connected_components(m.elm.tag1 == label_skin)
        idx = c.index(max(c, key=len))
        c.pop(idx)
        c = np.array([item for row in c for item in row]) - 1
        m.elm.tag1[c] = label_new
    
    m.elm.tag2[:] = m.elm.tag1
    return m


def _get_closest_skin_pos(pos, m, label_skin = 1005):
    ''' returns the position on the skin that is closest to the
        CoG of the provided positions
    '''
    CoG = np.mean(pos, axis = 0)
    idx_skin = np.where(m.elm.tag1 == label_skin)[0]
    elm_centers = m.elements_baricenters().value[idx_skin]
    distQ=np.sum((elm_centers - CoG)**2,axis=1)
    return elm_centers[np.argmin(distQ)]


def _get_optimal_center_pos(mask_pos, rest_pos, m, label_GM = 2, label_skin = 1005):
    ''' returns the position on the skin that has the optimal ratio between 
        resistance to mask and resistance to rest of GM
    '''
    mask_tets = m.find_tetrahedron_with_points(mask_pos, compute_baricentric=False)
    mask_tets = mask_tets[mask_tets != -1]
    mask_tets = mask_tets[m.elm.tag1[mask_tets-1] == label_GM]
    mask_nodes = np.unique(m.elm.node_number_list[mask_tets-1,:3])
    
    rest_tets = m.find_tetrahedron_with_points(rest_pos, compute_baricentric=False)
    rest_tets = rest_tets[rest_tets != -1]
    rest_tets = rest_tets[m.elm.tag1[rest_tets-1] == label_GM]
    rest_nodes = np.unique(m.elm.node_number_list[rest_tets-1,:3])
    rest_nodes = np.setdiff1d(rest_nodes, mask_nodes)
    
    cond_L = cond.standard_cond()
    cond_list = [c.value for c in cond_L]
    elm_cond = cond.cond2elmdata(m, cond_list)
    
    bcs = [fem.DirichletBC(mask_nodes, np.ones_like(mask_nodes, dtype=float)),
           fem.DirichletBC(rest_nodes, np.zeros_like(rest_nodes, dtype=float))]
    bc = fem.DirichletBC.join(bcs)
    
    S = fem.FEMSystem(m, elm_cond, dirichlet=bc)
    v = S.solve()
    v = mesh_io.NodeData(v, name='v', mesh=m)
    
    v = v.node_data2elm_data()
    m.add_element_field(v,'v')
    
    m = m.crop_mesh(elm_type = 2)
    m = m.crop_mesh(tags = label_skin)
    
    center_pos = m.elements_baricenters().value
    center_pos = center_pos[np.argmax(m.field['v'].value),:]
    
    del S
    gc.collect()
    return center_pos


def get_center_pos(m_surf, subject_path, condition, el_name=None):
    """
    returns the position of the center electrode on the 
    skin surface

    Parameters
    ----------
    m_surf : mesh_io.Msh
        mesh created by simnibs_memoslap_utils.get_central_gm_with_mask
    subject_path : string
        m2m-folder
    condition : string
        'closest': position with smallest distance to mask
        'optimal': position with optimal ratio between resistance to mask and
                    resistance to rest of GM
        'elpos': electrode position from EEG10-10 system
    el_name : string
        electrode name for condition 'elpos' (default: None)
            
    Returns
    -------
    pos_center : np.array
       position of center electrode

    """
    
    subject_files = SubjectFiles(subpath=subject_path)  
    
    # load head mesh
    m = mesh_io.read_msh(subject_files.fnamehead)
    if isSimNIBS4:
        # relabel skin in internal air cavities
        m = _relabel_internal_air(m)
        
    mask_pos = m_surf.nodes.node_coord[m_surf.field['mask'].value,:]
    rest_pos = m_surf.nodes.node_coord[~m_surf.field['mask'].value,:]
    
    if condition == 'closest':
        pos_center = _get_closest_skin_pos(mask_pos, m)
    elif condition == 'optimal':
        pos_center = _get_optimal_center_pos(mask_pos, rest_pos, m)
    elif condition == 'elpos':
        eeg_pos = sim_struct._get_eeg_positions(subject_files.get_eeg_cap())
        pos_center = eeg_pos[el_name]
    else:
        raise ValueError(f"unknown condition: {condition}")
        
    return pos_center



def _get_cereb_mask(subj_files, label_WM = 1, label_GM = 2, cereb_labels = [7, 8, 46, 47]):
    """
    returns volume mask that ends approx in the middle of the cerebellum GM
    using the charm results in the m2m-folder 

    Parameters
    ----------
    subj_files : file_finder.SubjectFiles
        SubjectFiles object with the subject-specific path and file names
    label_WM : int
        WM label in tissue_labeling_upsampled.nii.gz. The default is 1.
    label_GM : TYPE, optional
        GM label in tissue_labeling_upsampled.nii.gz. The default is 2.
    cereb_labels : list of int, optional
        all cerebellum labels in labeling.nii.gz. The default is [7, 8, 46, 47].

    Returns
    -------
    cereb_roi : numpy.ndarray
        3D volume mask (float)
    cereb_affine: numpy.ndarray
        affine matrix

    """
    # tissue labels at 0.5 mm resolution used for meshing 
    im_tissue_label = nib.load(subj_files.tissue_labeling_upsampled)
    buffer_tissue_label = np.round(im_tissue_label.get_fdata()).astype(np.uint16)
    
    # original (noisy) labels
    im_cereb = nib.load(subj_files.labeling)
    buffer_cereb = np.round(im_cereb.get_fdata()).astype(np.uint16)
    
    # extract cerebellum from original labels and upsample to 0.5 mm
    buffer_cereb = np.in1d(buffer_cereb,cereb_labels).reshape(buffer_cereb.shape)
    buffer_cereb = buffer_cereb.astype(np.uint16)
    buffer_cereb = transformations.volumetric_affine(
        (buffer_cereb, im_cereb.affine),
        np.eye(4),
        im_tissue_label.affine,
        im_tissue_label.shape,
        intorder=0
    )
    
    # multiply raw cerebellum mask with gm and wm from im_upsampled_tissue_seg
    # and close a bit
    se = ndimage.generate_binary_structure(3, 3)
    se_n = ndimage.generate_binary_structure(3, 1)
    
    buffer_cereb *= (buffer_tissue_label == label_GM) + (buffer_tissue_label == label_WM)    
    buffer_cereb = mrph.binary_erosion(buffer_cereb, se_n, 2)
    buffer_cereb = _get_largest_components(buffer_cereb, se, num_limit=1)
    buffer_cereb = mrph.binary_dilation(buffer_cereb, se_n, 2)
                             
    # get cerebellar middle gm in cropped volume
    buffer_wm = buffer_cereb * (buffer_tissue_label == label_WM)
    buffer_wm, cropped_affine, _ = transformations.crop_vol(buffer_wm, im_tissue_label.affine, buffer_cereb, thickness_boundary=2)
    buffer_cereb, _ , _ = transformations.crop_vol(buffer_cereb, im_tissue_label.affine, buffer_cereb, thickness_boundary=2)
    
    buffer_gm = buffer_cereb * ~buffer_wm
    buffer_middle = np.zeros(buffer_cereb.shape,dtype='float32')
    
    for i in range(1000):
        buffer_middle[buffer_wm] = 1.0
        buffer_middle[~buffer_cereb] = 0.0 
        buffer_middle_new = uniform_filter(buffer_middle, 3)
        delta = np.sum((buffer_middle_new[buffer_gm] - buffer_middle[buffer_gm])**2)/np.sum(buffer_middle_new[buffer_gm]**2)
        buffer_middle = buffer_middle_new
        if delta < 1e-9: 
            break
    
    return buffer_middle, cropped_affine


def create_cereb_surface(subpath,
                         label_WM = 1,
                         label_GM = 2,
                         cereb_labels = [7, 8, 46, 47],
                         level = 0.5,
                         force_overwrite = False
                         ):
    """
    Reconstructs a very coarse middle cerebellar surface, and stores it as
    'cerebellum.central.gii' in the 'surfaces'-subfolder of the m2m-folder

    Parameters
    ----------
    subpath : string
        m2m-folder
    label_WM : int, optional
        WM label in tissue_labeling_upsampled.nii.gz. The default is 1.
    label_GM : int, optional
        GM label in tissue_labeling_upsampled.nii.gz. The default is 2.
    cereb_labels : list of int, optional
        all cerebellum labels in labeling.nii.gz. The default is [7, 8, 46, 47].
    level : float, optional
        cut off value for surface creation (range 0 to 1). Higher values move 
        the surface closer to white matter. The default is 0.5.
    force_overwrite : bool, optional
        If set to True, any existing cerebellum.central.gii file will be
        overwritten. If set to False, the re-creation will be skipped.
        The default is False.

    Returns
    -------
    None.

    """
    subj_files = SubjectFiles(subpath = subpath)
    fname_out = os.path.join(subj_files.surface_folder, 
                             'cerebellum.central.gii')
    
    if (not force_overwrite) and os.path.exists(fname_out):
        logger.info(f'Found {fname_out}')
        logger.info('skipping re-creation...')
        return
    
    # make volume mask that ends approx in the middle of the cerebellum GM
    cereb_roi, cereb_affine = _get_cereb_mask(subj_files, 
                                              label_WM = label_WM,
                                              label_GM = label_GM,
                                              cereb_labels = cereb_labels
                                              )        
    # reconstruct surface
    m, _ = marching_cube(cereb_roi, 
                        affine=cereb_affine, 
                        level=level,
                        step_size=1.0,
                        only_largest_component=True,
                        n_uniform=2
                        )
    # keep only surface parts that are in GM
    im_tissue_label = nib.load(subj_files.tissue_labeling_upsampled)
    buffer_tissue_label = np.round(im_tissue_label.get_fdata()).astype(np.uint16)
    ed = mesh_io.ElementData.from_data_grid(m, 
                                            buffer_tissue_label, 
                                            im_tissue_label.affine
                                            )
    m = m.crop_mesh(elements = np.where(ed.value == label_GM)[0]+1)
    # keep only largest component
    components = m.elm.connected_components()
    components.sort(key=len,reverse=True)
    m = m.crop_mesh(elements=components[0])
    
    mesh_io.write_gifti_surface(m, fname_out)
    