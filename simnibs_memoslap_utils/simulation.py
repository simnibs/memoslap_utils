# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:28:45 2023

@author: axthi
"""

import os
import numpy as np
from copy import deepcopy

from .simu_settings import get_simu_template

from simnibs import mesh_io, sim_struct, __version__
from simnibs.utils.file_finder import SubjectFiles
from simnibs.utils.simnibs_logger import logger

version = [int(x) for x in __version__.split('.')[:3]]
isSimNIBS4 = version[0]>3
isSimNIBS4xx = (version[1]>0 or version[2]>0) and isSimNIBS4
isSimNIBS402 = (version[1]>0 or (version[1]==0 and version[2]>1)) and isSimNIBS4
if isSimNIBS4:
    from simnibs.utils import transformations
    from simnibs.utils.file_finder import get_reference_surf
else:
    from simnibs.msh import transformations
    from simnibs.utils.file_finder import templates
    from .extra_stuff_version3 import expand_to_center_surround
    

def _create_results_path(results_basepath, project_nr, exp_cond, subpath):
    ''' generate main results path
    '''
    subject_files = SubjectFiles(subpath=subpath)
    results_path = os.path.abspath(os.path.join(results_basepath, 
                                                'P'+str(project_nr)+'_'
                                                +exp_cond+'_'
                                                +subject_files.subid
                                                )
                                   )
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    return results_path


def _fitPlaneLTSQ(XYZ):
    ''' fit plane to point cloud to get normal vector
    '''
    (rows, cols) = XYZ.shape
    G = np.ones((rows, 3))
    G[:, 0] = XYZ[:, 0]  #X
    G[:, 1] = XYZ[:, 1]  #Y
    Z = XYZ[:, 2]
    (a, b, c),_,_,_ = np.linalg.lstsq(G, Z, rcond = None)
    normal = (-a, -b, 1)
    normal = normal / np.linalg.norm(normal)
    return c, normal


def _get_ydir_zdir(ff):
    ''' get vector pointing "upwards", based on the fitted eeg cap
    '''
    el_list = ['Fpz', 'Fp2', 'AF8', 'F8', 'FT8', 'T8', 'TP8', 'P8', 
               'PO8', 'O2', 'Oz', 'Fp1', 'AF7', 'F7', 'FT7', 'T7', 
               'TP7', 'P7', 'PO7', 'O1']
        
    eeg_pos = sim_struct._get_eeg_positions(ff.get_eeg_cap())
    pos_list = np.zeros((len(el_list),3))
    for i, el_name in enumerate(el_list):
        pos_list[i] = eeg_pos[el_name]    
    
    _, zdir = _fitPlaneLTSQ(pos_list)
    
    ydir = eeg_pos['Fz'] - eeg_pos['Cz']
    xdir = np.cross(ydir,zdir)
    xdir = xdir / np.linalg.norm(xdir)
    ydir = np.cross(zdir,xdir)
    
    return ydir, zdir


def _get_skin_normal(pos, ff, label_skin = 1005):
    ''' returns the normal direction on the skin node closest to the
        provided position
    '''
    m = mesh_io.read_msh(ff.fnamehead)
    m = m.crop_mesh(tags=label_skin)
    m = m.crop_mesh(elm_type=2)

    distQ=np.sum((m.nodes.node_coord - pos)**2, axis=1)
    idx_node = np.argmin(distQ)
        
    nd = m.nodes_normals()
    norm_vec = nd.value[idx_node]
    return norm_vec



def _prep_simu(subpath, results_path, project, pos_center):
    ''' prepare the SESSION sim_struct for the FEMs.
        this includes defining the surround electrode positions
    '''
    ff = SubjectFiles(subpath=subpath)
    S, EL_surround = get_simu_template()
    S.subpath = subpath
    S.pathfem = os.path.join(results_path, 'raw_results')
    if not isSimNIBS4:
        S.fnamehead = ff.fnamehead
            
    tdcs_list_template = S.poslists[0]
    S.poslists=[]
    tdcs_list_template.currents = project.current
    tdcs_list_template.electrode[0].centre = pos_center
    phis_surround = 360*np.arange(project.N_surround)/project.N_surround
    radii = [project.radius] if not isinstance(project.radius, list) else project.radius
    
    ydir, zdir = _get_ydir_zdir(ff)
    norm_vec = _get_skin_normal(pos_center, ff)
    if np.sum(zdir*norm_vec) < 0.9: # approx. 20 deg
        offset = zdir
    else:
        offset = ydir
        logger.info('using ydir as reference direction for surround electrode positions ...')
            
    for r in radii:
        tdcs_list = deepcopy(tdcs_list_template)
        if isSimNIBS4:
            tdcs_list.expand_to_center_surround(S.subpath, r, project.N_surround,
                                                pos_dir_1stsurround = pos_center + 20*offset,
                                                multichannel=True,
                                                phis_surround = phis_surround+project.phi,
                                                el_surround = EL_surround)
        else:
            tdcs_list = expand_to_center_surround(tdcs_list, r, project.radius, project.N_surround,
                                      pos_dir_1stsurround = pos_center + 20*offset,
                                      multichannel=True,
                                      phis_surround = phis_surround+project.phi,
                                      el_surround = EL_surround)
        S.add_tdcslist(tdcslist=tdcs_list)

    return S


def _extract_surround_pos(S, project):
    pos_surround = {}
    for i, tdcslist in enumerate(S.poslists):
        el_pos = np.empty((len(tdcslist.electrode)-1,3), dtype=np.float64)
        for k, el in enumerate(tdcslist.electrode[1:]):
            el_pos[k] = el.centre
        pos_surround[project.radius[i]] = el_pos
    return pos_surround


def _calc_quantities(nd, quantities):
    d = dict.fromkeys(quantities)
    for q in quantities:
        if q == 'magn':
            d[q] = nd.norm()
        elif q == 'normal':
            d[q] = nd.normal()
            d[q].value *= -1
        elif q == 'tangent':
            d[q] = nd.tangent()
        elif q == 'angle':
            d[q] = nd.angle()
        else:
            raise ValueError('Invalid quantity: {0}'.format(q))
    return d


def _map_E_to_surf(res_list, m_surf, results_path,
                  quantities=['magn', 'normal', 'tangent']):
    ''' map E from the volume meshes to the GM centeral surfaces
    '''
    quantities=['magn', 'normal', 'tangent']

    res_list_surf = dict()
    for radius, fname_msh in res_list.items():        
        m = mesh_io.read_msh(fname_msh)
        # Crop out WM, GM, and CSF. We add WM and CSF to make the mesh convex.
        m = m.crop_mesh(tags=[1, 2, 3])
        # Set the volume to be GM. The interpolation will use only the tetrahedra in the volume.
        th_indices = m.elm.elm_number[m.elm.tag1 == 2]
        
        m_results = deepcopy(m_surf)
        nd = m.field['E'].interpolate_to_surface(m_results, th_indices=th_indices)
        q = _calc_quantities(nd, quantities)
    
        for q_name, q_data in q.items():
            m_results.add_node_field(q_data, 'E_' + q_name)
    
        fname_out = os.path.join(results_path,
                                 os.path.basename(results_path)+'_'+str(int(radius))+'.msh')
        mesh_io.write_msh(m_results, fname_out)
        res_list_surf[radius] = fname_out
        
    return res_list_surf


def _write_visualizations(subpath, results_path, res_list, pos_center, pos_surround):
    ''' write out the .geo and .opt files
    '''
    label_skin = 1005
    
    subject_files = SubjectFiles(subpath=subpath)  
    
    # save skin from head mesh as .geo-file in results dir
    m = mesh_io.read_msh(subject_files.fnamehead)
    idx = (m.elm.tag1 == label_skin) & (m.elm.elm_type == 2)
    fname_scalp = os.path.join(results_path, os.path.basename(results_path)+'_skin.geo')
    mesh_io.write_geo_triangles(m.elm[idx, :3]-1, m.nodes.node_coord,
                                fname_scalp, name='scalp')
    
    # add .opt-files to surface gm meshes
    for radius, fname_msh in res_list.items():
        # .geo-file with electrode positions
        fname_geo = os.path.splitext(fname_msh)[0]+'_elpos.geo'    
        pos = np.vstack((pos_center.reshape((1,3)), pos_surround[radius]))
        values = np.zeros((pos.shape[0]), float)
        values[0] = 1
        mesh_io.write_geo_spheres(pos, fname_geo, values = values, name='electrode_pos')
        
        # .opt-fiel with visualization settings
        m = mesh_io.read_msh(fname_msh)
        v = m.view(visible_fields = 'E_magn')
        v.add_merge(fname_geo)
        v.add_merge(fname_scalp)
        v.add_view(ColormapNumber=14, Visible=1, ShowScale=0)  # el-pos
        v.add_view(ColormapNumber=8, ColormapAlpha=.3,
                   Visible=1, ShowScale=0)  # scalp
        v.write_opt(fname_msh)
    

def _get_templates_and_trafos(subpath):
    ''' load fsaverage surfaces and trafos to fsaverage space
    '''
    subject_files = SubjectFiles(subpath=subpath)
    
    # get filenames
    fn_sphere = {}
    fn_avg_central = {}
    fn_reg = {}
    for hemi in ['lh', 'rh']:
        if isSimNIBS4:
            fn_sphere[hemi] = get_reference_surf(hemi, 'sphere')
            fn_avg_central[hemi] = get_reference_surf(hemi, 'central')
            fn_reg[hemi] = subject_files.get_surface(hemi, 'sphere_reg')
        else:
            if hemi == 'lh':
                fn_sphere[hemi] = templates.cat_lh_sphere_ref
                fn_avg_central[hemi] = templates.cat_lh_cortex_ref
                fn_reg[hemi] = subject_files.lh_reg
            else:
                fn_sphere[hemi] = templates.cat_rh_sphere_ref
                fn_avg_central[hemi] = templates.cat_rh_cortex_ref
                fn_reg[hemi] = subject_files.rh_reg
    
    # load surfaces
    ref_surf = {}
    avg_surf = {}
    reg_surf = {}
    for hemi in ['lh', 'rh']:
        ref_surf[hemi] = mesh_io.read_gifti_surface(fn_sphere[hemi])
        avg_surf[hemi] = mesh_io.read_gifti_surface(fn_avg_central[hemi])
        reg_surf[hemi] = mesh_io.read_gifti_surface(fn_reg[hemi])
                    
    return ref_surf, avg_surf, reg_surf


def _map_results_to_fsavg(subject_path,res_list,out_folder):
    ''' map the results from the meshes stored in res_list to fsaverage space
    '''
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
        
    ref_surf, avg_surf, reg_surf = _get_templates_and_trafos(subject_path)
    
    # write fsavg central surfaces to out_folder (for convenience)
    for hemi in ['lh', 'rh']:
        if isSimNIBS402:
            mesh_io.write_freesurfer_surface(
                avg_surf[hemi],
                os.path.join(out_folder, hemi + '.fsavg.central')
            )
        else:
            mesh_io.write_freesurfer_surface(
                avg_surf[hemi],
                os.path.join(out_folder, hemi + '.fsavg.central'),
                ref_fs=True
            )
    
    # join hemis of fsavg template
    hemi_idx = {'lh': 1001, 'rh': 1002}
    avg_joined = mesh_io.Msh()
    for hemi in ['lh', 'rh']:    
        avg_surf[hemi].elm.tag1[:] = hemi_idx[hemi]
        avg_joined = avg_joined.join_mesh(avg_surf[hemi])
    avg_joined.elm.tag2[:] = avg_joined.elm.tag1
    
    # convert results to fsavg space
    for radius, fname_msh in res_list.items():
        m = mesh_io.read_msh(fname_msh)
        
        # make dict for storing the joined data after transformation
        data_joined = {}
        for name, data in m.field.items():    
            shape = list(data.value.shape)
            shape[0] = 0
            data_joined[name] = np.empty(shape, data.value.dtype)
        
        # transform and store as freesurfer curvature files
        for hemi in ['lh', 'rh']:
            fn_out = os.path.splitext(os.path.basename(fname_msh))[0]
            fn_out = os.path.join(out_folder, hemi+'.fsavg.'+fn_out+'.')
            
            m_hemi = m.crop_mesh(tags = hemi_idx[hemi])
            assert m_hemi.elm.nr == reg_surf[hemi].elm.nr
            
            if isSimNIBS4xx:
                morph = transformations.SurfaceMorph(reg_surf[hemi], 
                                                     ref_surf[hemi], 
                                                     method="linear")
            else:
                kdtree = None
                
            for name, data in m_hemi.field.items():
                if isSimNIBS4xx:
                    data_fsavg = morph.transform(data.value)
                else:
                    data_fsavg, kdtree = transformations._surf2surf(data.value,
                                                          reg_surf[hemi],
                                                          ref_surf[hemi],
                                                          kdtree
                                                          )
                mesh_io.write_curv(fn_out+name, data_fsavg, ref_surf[hemi].elm.nr)
                data_joined[name] = np.append(data_joined[name], data_fsavg, axis=0)
        
        # write fsavg-transformed data (combined lh and rh) as .msh-file
        msh_out = deepcopy(avg_joined)
        for name, data in data_joined.items():
            assert avg_joined.nodes.nr == data.shape[0]
            msh_out.add_node_field(data, name)
        fn_out = os.path.splitext(os.path.basename(fname_msh))[0]
        fn_out = os.path.join(out_folder, 'fsavg.'+fn_out+'.msh')
        mesh_io.write_msh(msh_out, fn_out)
    

def run_FEMs(subject_path, project, results_basepath, pos_center, m_surf, map_to_fsavg=False):
    """ performs the following steps:
        * set up the FEM simulation (includes defining the surround 
                                     electrode positions)
        * run the FEM simulations
        * map results onto the middle GM surfaces
        * map results of lh and rh to fsaverage (optional)
        
    Parameters
    ----------
    subject_path : string
        m2m-folder.
    project : simnibs_memoslap_utils.project_settings.project_template
        project settings.
    results_basepath : string
        folder to which the results will be added as subfolder.
    pos_center : np.array
        position of center electrode.
    m_surf: simnibs.mesh_io.Msh
        central gm surfaces created by get_central_gm_with_mask   
    map_to_fsavg : bool, optional
        set to True for transforming the results to fsaverage space
        (only lh and rh). The standard is False.

    Returns
    -------
    pos_surround : dict
        dictionary with the surround electrode positions for each radius
    results_path : string
        main results path.
    res_list : dict
        pathnames to the result meshes (on individual GM surfaces) for each radius
    res_list_raw : dict
        dictionary with the pathnames to the raw result meshes

    """
    results_path = _create_results_path(results_basepath,
                                        project.proj_nr,
                                        project.exp_cond,
                                        subject_path) 
    
    # prepare simus
    logger.info('Setting up simulations and getting surround electrode positions ...')
    S = _prep_simu(subject_path, results_path, project, pos_center)
    pos_surround = _extract_surround_pos(S, project)
    
    # run the FEMs
    logger.info('Running FEMs ...')
    fname_rawresults = S.run()
    res_list_raw = dict(zip(project.radius, fname_rawresults))
    
    # map E-fields to the central gm surfaces
    logger.info('Mapping E-field to central gm surfaces ...')
    res_list = _map_E_to_surf(res_list_raw, m_surf, results_path)
    _write_visualizations(subject_path, results_path, res_list, pos_center, pos_surround)

    # map E-fields to lh and rh in fsaverage space
    if map_to_fsavg:
        logger.info('Mapping lh and rh results to fsaverage space ...')
        _map_results_to_fsavg(subject_path,
                              res_list,
                              os.path.join(results_path,'fsavg')
                              )
    
    return pos_surround, results_path, res_list, res_list_raw


def analyse_simus(res_list):
    """
    calculate the medians in the mask and the focalities for each radius
    and each field quantity (magn, normal, tangent)

    Parameters
    ----------
    res_list : dict
        pathnames to the result meshes (on individual GM surfaces) for each radius

    Returns
    -------
    res_summary : dict
        dictionary with the focalities and field medians in the mask for all
        radii. For example, to print the tested radii with the corresponding
        median and focality values:
        print(res_summary['radius'])
        print(res_summary['roi_median']['E_magn'])
        print(res_summary['focality']['E_magn'])

    """
    roi_median = {}
    focality = {}
    roi_squared = {}
    #roi_surround_squared_diff = {}
    for radius, fname_msh in res_list.items():        
        m = mesh_io.read_msh(fname_msh)
        nd_sze = m.nodes_volumes_or_areas().value
        idx_mask = m.field['mask'].value > 0
        
        res_quantities = list(m.field.keys())
        res_quantities.remove('mask')
        for q in res_quantities:
            roi_med = np.median(m.field[q].value[idx_mask])
            foc = np.sum(nd_sze[ m.field[q].value > roi_med ])
            roi_sq = np.sum(m.field[q].value[idx_mask]**2)
            #roi_sur_sq = np.sum(m.field[q].value[idx_mask]**2) - np.sum(m.field[q].value[~idx_mask]**2)
            if not q in roi_median:
                roi_median[q] = [roi_med]
                focality[q] = [foc]
                roi_squared[q] = [roi_sq]
                #roi_surround_squared_diff[q] = [roi_sur_sq]
            else:
                roi_median[q].append(roi_med)
                focality[q].append(foc)
                roi_squared[q].append(roi_sq)
                #roi_surround_squared_diff[q].append(roi_sur_sq)
                    
    res_summary = {'radius': list(res_list.keys()),
                   'roi_median': roi_median,
                   'focality': focality,
                   'roi_squared': roi_squared,
                   #'roi_surround_squared_diff': roi_surround_squared_diff
                   }
    return res_summary 
