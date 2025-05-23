# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 22:58:25 2023

@author: axthi
"""

import re
import os
import logging
import pickle

from .preparation import create_cereb_surface, get_central_gm_with_mask, get_center_pos
from .simulation import run_FEMs, analyse_simus, _create_results_path
from .reporting import internal_report, placement_guide

from simnibs import __version__
from simnibs.utils import simnibs_logger
from simnibs.utils.simnibs_logger import logger

import datetime

version = [int(x) for x in __version__.split('.')[:3]]
isSimNIBS4 = version[0]>3
isSimNIBS4xx = (version[1]>0 or version[2]>0) and isSimNIBS4
isSimNIBS402 = (version[1]>0 or (version[1]==0 and version[2]>1)) and isSimNIBS4
if isSimNIBS4:
    from .write_nnav import write_nnav_files


def _setup_logger(logfile):
    """Add FileHandler etc."""
    with open(logfile, "a") as f:
        f.write("<HTML><HEAD><TITLE>simnibs memoslap run</TITLE></HEAD><BODY><pre>")
        f.close()
    fh = logging.FileHandler(logfile, mode="a")
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    simnibs_logger.register_excepthook(logger)


def _stop_logger(logfile):
    """Close down logging"""
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
    simnibs_logger.unregister_excepthook()
    logging.shutdown()
    with open(logfile, "r") as f:
        logtext = f.read()

    # Explicitly remove this really annoying stuff from the log
    removetext = (
        re.escape("-\|/"),
        re.escape("Selecting intersections ... ")
        + "\d{1,2}"
        + re.escape(" %Selecting intersections ... ")
        + "\d{1,2}"
        + re.escape(" %"),
    )
    with open(logfile, "w") as f:
        for text in removetext:
            logtext = re.sub(text, "", logtext)
        f.write(logtext)
        f.write("</pre></BODY></HTML>")
        f.close()


def run(subject_path, project, results_basepath='.', add_cerebellum=True, map_to_fsavg=False):
    """
    wrapper function around the following steps:
        * create a coarse cerebellum central gm surface and add
          it to the m2m-folder content (only for charm results)
        * map mask to middle GM surfaces
        * get positions of center electrode
        * get the surround electrode positions and run FEM
        * map e-field onto the middle GM surfaces
        * map e-field of lh and rh to fsaverage (optional)
        * export electrode positions for use with neuronavigation (only simnibs4)

    Parameters
    ----------
    subject_path : string
        m2m-folder.
    project : simnibs_memoslap_utils.project_settings.project_template
        project settings.
    results_basepath : string, optional
        folder to which the results will be added as subfolder. The default is '.'.
    add_cerebellum : bool, optional
        DESCRIPTION. The default is True.
    map_to_fsavg : bool, optional
        set to True for transforming the results to fsaverage space
        (only lh and rh). The standard is False.

    Returns
    -------
    res_list : dict
        pathnames to the result meshes (on individual GM surfaces) for each radius
    res_list_raw : dict
        dictionary with the pathnames to the raw result meshes
    pos_center : np.array
        position of center electrode.
    pos_surround : dict
        dictionary with the surround electrode positions for each radius
    res_summary : dict
        dictionary with the E-field medians in the mask and the focalities

    """
    # create results path and start logger
    results_path = _create_results_path(results_basepath,
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

    # perform the following steps:
    #   * set up the FEM simulation (sets also the surround electrode positions)
    #   * run the FEM simulations
    #   * map results onto the middle GM surfaces
    #   * map results of lh and rh to fsaverage (optional)
    logger.info('Starting FEM part...')
    [pos_surround, results_path,
     res_list, res_list_raw] = run_FEMs(subject_path,
                                         project,
                                         results_basepath,
                                         pos_center,
                                         m_surf,
                                         map_to_fsavg
                                        )

    # get field medians and focality
    logger.info('Calculating field medians in mask and focalities ...')
    res_summary = analyse_simus(res_list)

    # export electrode positions for use with neuronavigation (only simnibs4)
    if isSimNIBS4:
        logger.info('Exporting positions for neuronavigation systems...')
        write_nnav_files(subject_path, res_list, pos_center, pos_surround)
    else:
        logger.warning('Running SimNIBS 3: No position export for nnav possible.')

    # saving results and stopping
    fn_out = os.path.join(results_path,'simnibs_memoslap_results.pkl')
    with open(fn_out, 'wb') as filedump:
        pickle.dump([project.asdict(),
                     pos_center,
                     pos_surround,
                     res_list,
                     res_list_raw,
                     res_summary,
                     __version__[0],
                     datetime.datetime.now()
                    ], filedump)

    # write report pdf files
    for radius, fname_msh in res_list.items():
        results_path = os.path.split(fname_msh)[0]
        internal_report(results_path, radius)
        placement_guide(results_path, radius)
        placement_guide(results_path, radius,
                        nexstim=True, subject_path=subject_path)

    logger.info('=====================================')
    logger.info(f'SimNIBS Memoslap run finished for {subject_path}')
    logger.info(f'Results are in {results_path}')
    logger.info(' ')
    logger.info('Radii (in mm), median E-field magnitudes (in V/m) and focalities (in mm²):')
    logger.info('  '.join( ['{:.1f} '.format(elem) for elem in res_summary['radius']] ))
    logger.info('  '.join( ['{:.3f}'.format(elem) for elem in res_summary['roi_median']['E_magn']] ))
    logger.info('  '.join( ['{:.0f} '.format(elem) for elem in res_summary['focality']['E_magn']] ))
    logger.info('=====================================')

    _stop_logger(logfile)

    return res_list, res_list_raw, pos_center, pos_surround, res_summary
