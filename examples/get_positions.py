# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 11:12:54 2023

@author: axthi
"""

import sys
import argparse

path_to_utils = 'C:/Users/axthi/simnibsN/memoslap/' # NOTE: change to your memoslap git directory
sys.path.append(path_to_utils)

import simnibs_memoslap_utils as smu 


def _get_positions(project_nr, subject_path, results_basepath,
                   add_cerebellum=True, map_to_fsavg=True):
    """ run target and control simulations for the given subject and project
    """
    project = smu.projects[project_nr]['target']
    smu.run(subject_path, project, results_basepath,
            add_cerebellum=add_cerebellum,
            map_to_fsavg=map_to_fsavg)

    project = smu.projects[project_nr]['control']
    smu.run(subject_path, project, results_basepath,
            add_cerebellum=add_cerebellum,
            map_to_fsavg=map_to_fsavg)

def main():
    parser = argparse.ArgumentParser(prog='get_positions')
    parser.add_argument("--subjpath", type=str, required=True,
                         help="m2m-folder created by charm")
    parser.add_argument("--proj", type=int, required=True,
                         help="Project number (1-8)")
    parser.add_argument("--outpath", type=str, required=True,
                         help="results will be added as subfolder to the given path")
    parser.add_argument("--add_cerebellum", type=str, default="true",
                        help="add cerebellum central GM surface (default: true)")
    parser.add_argument("--map_to_fsavg", type=str, default="true",
                        help="map simulation results to fsaverage space (default: true)")
    opt = parser.parse_args()
    
    add_cerebellum = opt.add_cerebellum.lower() == 'true'
    map_to_fsavg= opt.map_to_fsavg.lower() == 'true'
    
    _get_positions(opt.proj, opt.subjpath, opt.outpath,
                   add_cerebellum=add_cerebellum, 
                   map_to_fsavg=map_to_fsavg)

if __name__ == '__main__':
    main()
