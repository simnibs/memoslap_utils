# -*- coding: utf-8 -*-
"""
Created on 16.5.2023

@author: jevrih

Command line utility for preparing summary reports of stimulations and guides
for electrode placement. The only thing the user needs to change is the
directory in the sys.path.append() a few lines below. Use the command line
parameters described below to specify your report. Parameters "report",
"subj_path," "proj," "subj," and "radius" are required, and the rest should
not be necessary as long as you are doing the standard analyses.

"""

import sys
import os
from os.path import join
sys.path.append("/home/jev/memoslap") # change to your memoslap directory

import argparse
import simnibs_memoslap_utils as smu
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--report", type=str, required=True,
                     help="Type of report: 'placement' 'internal' or 'both'")
parser.add_argument("--subj_path", type=str, required=True,
                     help="Directory where files are located")
parser.add_argument("--proj", type=int, required=True,
                     help="Project number (1-8)")
parser.add_argument("--subj", type=str, required=True,
                     help="Subject name")
parser.add_argument("--radius", type=int, required="",
                     help="Montage radius")
parser.add_argument("--coo_system", type=str, default="RAS",
                     help="RAS or LPS coordinate systems")
parser.add_argument("--condition", type=str, default="target",
                     help="Target or control. Only makes diff w/default radius")
parser.add_argument("--results_pkl", type=str,
                     default="simnibs_memoslap_results.pkl",
                     help="PKL file containing simulation results.")
parser.add_argument("--skin", type=str, default="",
                     help="Skin file. Attemps inference if unspecified.")
parser.add_argument("--mesh", type=str, default="",
                     help="Mesh file. Attemps inference if unspecified.")
parser.add_argument("--elpos", type=str, default="",
                     help="Electrode geo file. Attemps inference if unspecified.")
parser.add_argument("--elpos_bs", type=str, default="",
                     help="Electrode brainsight file. Attemps inference if unspecified.")
parser.add_argument("--out", type=str, default="",
                     help="Output pdf name. Builds from other paramters if unspecified.")
parser.add_argument("--cam_dist", type=float, default=500.,
                     help="Distance of camera from focal point.")
parser.add_argument("--alloc", type=str, default="Unspecified",
                     help="Which stimulation condition this represents.")
parser.add_argument("--cond_order", type=str, default="Unspecified",
                     help="Which condition order this subject had.")

opt = parser.parse_args()

# set up paths and filenames, check out input validity
if opt.report not in ["placement", "internal", "both"]:
    raise ValueError(f"Invalid 'report' parameter {opt.report}")
if opt.coo_system not in ["RAS", "LAS"]:
    raise ValueError(f"Invalid 'report' parameter {opt.report}")

if not opt.radius:
    radius = int(smu.projects[opt.proj][opt.condition].radius[0])
else:
    radius = opt.radius
if not opt.skin:
    skin_file = f"P{opt.proj}_{opt.subj}_skin.geo"
else:
    skin_file = opt.skin
if not opt.mesh:
    mesh_file = f"P{opt.proj}_{opt.subj}_{radius}.msh"
else:
    mesh_file = opt.mesh
if not opt.elpos:
    elpos_file = f"P{opt.proj}_{opt.subj}_{radius}_elpos.geo"
else:
    elpos_file = opt.elpos_file
if not opt.elpos_bs:
    elpos_bs_file = f"P{opt.proj}_{opt.subj}_{radius}" \
                    f"brainsight_{opt.coo_system}.txt"
else:
    elpos_bs_file = opt.elpos_bs

if opt.out:
    if opt.report == "both":
        raise ValueError("Cannot specify report as 'both' and specify 'out'")
    else:
        out_file = opt.out

# go
mesh_path = join(opt.subj_path, f"P{opt.proj}_{opt.subj}")
if opt.report == "placement" or opt.report == "both":
    if not opt.out:
        out_file = f"P{opt.proj}_{opt.subj}_{radius}_placement.pdf"
    smu.reporting.placement_guide(mesh_path, skin_file, elpos_file,
                                  elpos_bs_file, out_file, opt.coo_system,
                                  opt.proj, opt.subj, radius, opt.cam_dist)
if opt.report == "internal" or opt.report == "both":
    if not opt.out:
        out_file = f"P{opt.proj}_{opt.subj}_{radius}_internal.pdf"
    smu.reporting.internal_report(mesh_path, opt.results_pkl, skin_file,
                                  elpos_file, mesh_file, out_file,
                                  opt.proj, opt.subj, radius, opt.cam_dist,
                                  opt.alloc, opt.cond_order)
