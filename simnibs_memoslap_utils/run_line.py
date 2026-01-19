# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 19:03:57 2025

@author: axthi
"""
import os
import shutil
import numpy as np
import pickle
import datetime
from copy import deepcopy

from .preparation import create_cereb_surface, get_central_gm_with_mask, get_center_pos, _relabel_internal_air
from .simulation import _prep_simu, _extract_surround_pos, _map_E_to_surf, analyse_simus
from .run import _setup_logger, _stop_logger

from simnibs.utils.file_finder import SubjectFiles
from simnibs.utils.simnibs_logger import logger
from simnibs import __version__, mesh_io

version = [int(x) for x in __version__.split('.')[:3]]
isSimNIBS4 = version[0]>3


def _create_results_path_shifts(results_basepath, project_nr, exp_cond, subpath):
    ''' generate main results path for shift analyses
    '''
    subject_files = SubjectFiles(subpath=subpath)
    results_path = os.path.abspath(os.path.join(results_basepath, 
                                                'P'+str(project_nr)+'_'
                                                +exp_cond+'_'
                                                +subject_files.subid+'_shift'
                                                )
                                   )
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    return results_path


def _write_visualizations(subpath, results_path, res_list, idx_line, line, pos_shifts):
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
    
    fname_geo = os.path.join(results_path, os.path.basename(results_path)+'_line'+str(idx_line+1)+'.geo')
    mesh_io.write_geo_spheres(line, fname_geo, values = pos_shifts, name='line'+str(idx_line+1))
    
    fname_msh = os.path.join(results_path, os.path.basename(results_path)+'_line'+str(idx_line+1)+'.msh')
    for i, (_, fname) in enumerate(res_list.items()):
        m_hlp = mesh_io.read_msh(fname)
        if i == 0:
            m_out = deepcopy(m_hlp)
            m_out.nodedata = []
            m_out.add_node_field(m_hlp.field['mask'],'mask')
        m_out.add_node_field(m_hlp.field['E_magn'],'E_magn_l'+str(idx_line+1)+'_'+str(pos_shifts[i]))
    mesh_io.write_msh(m_out,fname_msh)
    
    v = m_out.view(visible_fields = 'E_magn_l'+str(idx_line+1)+'_'+str(pos_shifts[0]))
    cm = 0.
    for k in v.View[1:]:
        cm = max(cm,k.CustomMax)
    for k in v.View[1:]:
        k.CustomMax = cm
    v.add_merge(fname_geo)
    v.add_merge(fname_scalp)
    v.add_view(ColormapNumber=14, Visible=1)  # el-pos
    v.add_view(ColormapNumber=8, ColormapAlpha=.3, Visible=1, ShowScale=0)  # scalp
    v.write_opt(fname_msh)


def run_line(subject_path, project, results_basepath='.', shift_distance=30., shift_steps=7.5, add_cerebellum=True, fix_internal_air=True):
    """
    Analyses of the E-fields for montages systematically shifted along two orthogonal lines.
    Includes the following steps:
        * create a coarse cerebellum central gm surface and add
          it to the m2m-folder content (only for charm results)
        * map mask to middle GM surfaces
        * get positions of center electrode above ROI.
        * create two orthogonal lines of shifted center positions
        * get the surround electrode positions for all center positions along
          the two lines and run FEMs
        * map e-fields onto the middle GM surfaces
        * extract median of E-Field in ROI and focality in dependence of center position 

    Parameters
    ----------
    subject_path : string
        m2m-folder.
    project : simnibs_memoslap_utils.project_settings.project_template
        project settings.
    results_basepath : string, optional
        folder to which the results will be added as subfolder. The default is '.'.
    shift_distance : float, optional
        maximal distance the center position is shifted from its initial
        position above the ROI. Given in [mm]. The default is 30..
    shift_steps : float, optional
        Steps of the shifts in [mm]. The default is 7.5.
    add_cerebellum : bool, optional
        whether to add cerebellum surface to analyses. The default is True.
    fix_internal_air : bool, optional
        when True, SimNIBS4 head meshes in their m2m-folders will be replaced
        by versions in which internal air surfaces are relabled to 1099.
        The default is True.
    
    Returns
    -------
    res_list : dict
        dictionary with the simulation settings and results

    """
    assert len(project.radius) == 1
    
    # create results path and start logger
    results_path = _create_results_path_shifts(results_basepath,
                                            project.proj_nr,
                                            project.exp_cond,
                                            subject_path)
    logfile = os.path.join(results_path,'simnibs_memoslap_log.html')
    _setup_logger(logfile)    
    
    # create a coarse cerebellum central gm surface and
    # add it to the m2m-folder content (only for charm results)
    if add_cerebellum:
        logger.info('Creating cerebellum central gm surface...')
        create_cereb_surface(subject_path)
    
    # relabel internal air in SimNIBS4 head meshes
    if fix_internal_air and isSimNIBS4:
        logger.info("Relabeling internal air boundaries")
        subj_files = SubjectFiles(subpath = subject_path)
        m = mesh_io.read_msh(subj_files.fnamehead)
        
        if any(m.elm.tag1 == 1099):
            logger.info("internal air is already relabeled, skipping...")
        else:
            shutil.copyfile(subj_files.fnamehead, 
                            os.path.splitext(subj_files.fnamehead)[0] + '_org.msh')
            m = _relabel_internal_air(m)
            mesh_io.write_msh(m,subj_files.fnamehead)
    
    # load middle gm surfaces and add the mask as node data
    logger.info('Loading central gm surfaces and mapping mask onto surfaces...')
    m_surf = get_central_gm_with_mask(subject_path,
                                      project.hemi,
                                      project.fname_roi,
                                      project.mask_type,
                                      add_cerebellum
                                      )
    
    # get position of center electrode
    logger.info('Determining position of center elecrode...')
    pos_center = get_center_pos(m_surf, subject_path, project.condition, project.el_name)
    
    # get the shifted center electrode positions along two orthongonal lines
    logger.info('Determining shifted positions of center elecrode along two orthongonal lines...')
    proj_help = deepcopy(project)
    proj_help.N_surround = 4
    proj_help.phi = 0.
    proj_help.radius = list(np.arange(shift_steps, shift_distance+.1, shift_steps))
    S = _prep_simu(subject_path, '', proj_help, pos_center)
        
    assert len(S.poslists) == len(proj_help.radius)
    
    line1 = pos_center.reshape((1,3))
    line2 = pos_center.reshape((1,3))
    for i in range(len(S.poslists)):
        PL = S.poslists[i]
        line1 = np.vstack((PL.electrode[3].centre,line1,PL.electrode[1].centre))
        line2 = np.vstack((PL.electrode[4].centre,line2,PL.electrode[2].centre))
    lines = [line1, line2]
    
    # run simulations along the two lines
    res_line = []
    for idx_line, line in enumerate(lines):
        # prepare and run FEMs
        logger.info('Run FEMs for line '+str(idx_line+1)+' ...')
        list_pos_surround = list()
        for i, cp in enumerate(line):
            Shlp = _prep_simu(subject_path, results_path, project, cp)
            pos_surround = _extract_surround_pos(Shlp, project)
            list_pos_surround.append(pos_surround[list(pos_surround.keys())[0]])
            if i == 0: 
                S = Shlp
            else:
                S.add_tdcslist(Shlp.poslists[0])
    
        S.pathfem += '_line'+str(idx_line+1)
        fname_rawresults = S.run()
    
        # map results for line onto the middle GM surfaces
        logger.info('map results for line '+str(idx_line+1)+' onto the middle GM surfaces...')
        res_list_raw = dict(zip(np.arange(len(line))+1000*(idx_line+1)+1, fname_rawresults))
        res_list = _map_E_to_surf(res_list_raw, m_surf, results_path)
    
        # get field medians and focality
        logger.info('Calculating field medians in mask and focalities for line '+str(idx_line+1)+' ...')
        res_hlp = analyse_simus(res_list)
    
        # write common results mesh for this line
        pos_shifts = np.arange(-shift_distance, shift_distance+.1, shift_steps)
        _write_visualizations(subject_path, results_path, res_list, idx_line, line, pos_shifts)
        
        # assemble relevant results in one dict
        res_line.append({'pos_shifts': pos_shifts,
                         'center_pos': line1,
                         'surround_pos': list_pos_surround,
                         'fname_rawresults': fname_rawresults,
                         'fname_gmresults': res_list,
                         'roi_median': res_hlp['roi_median'],
                         'focality': res_hlp['focality'],
                         })
    
    # saving results and stopping
    fn_out = os.path.join(results_path,'simnibs_memoslap_results.pkl')
    with open(fn_out, 'wb') as filedump:
        pickle.dump([project.asdict(),
                     shift_distance,
                     shift_steps,
                     res_line,
                     __version__[0],
                     datetime.datetime.now()
                    ], filedump)
    
    _stop_logger(logfile)
    
    return res_line
    