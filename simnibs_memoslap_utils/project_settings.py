# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 19:56:02 2023

@author: axthi
"""

import os

my_path = os.path.dirname(os.path.realpath(__file__))


class project_template:
    def __init__(self):
        self.proj_nr = 0
        self.exp_cond = ''          # 'target' or 'control'
        self.roi = ''
        self.hemi = ''              # 'lh', 'rh' or 'cereb'
        self.mask_type = ''         # 'curv', 'mnivol'
        self.phi = 0.
        self.radius = [0.]
        self.el_name = ''

        # --- standard settings shared across projects:
        self.current = 0.002        # current flow through center channel (A)
        self.N_surround = 3         # number of surround electrodes
        self.condition = 'closest'  # 'closest', 'optimal' or 'elpos'
        # ---

        self.fname_roi = ''         # do not change; auto-generated path to file

    def __setattr__(self, name, value):
        if name == 'radius':
             self.__dict__[name] = [value] if not isinstance(value, list) else value
        else:
             self.__dict__[name] = value

        if name == 'roi':
            self.fname_roi = os.path.join(my_path,'masks',self.roi)

    def __repr__(self):
        return str(vars(self))

    def asdict(self):
        return vars(self)


# definition of TARGET conditions
projects = dict()

# P1
p = project_template()
p.proj_nr = 1
p.exp_cond = 'target'
p.roi = 'P1_RH_OTC'
p.hemi = 'rh'
p.mask_type = 'curv'
p.phi = 35.
p.radius = 40.
projects.update({p.proj_nr: {p.exp_cond: p}})

# P2
p = project_template()
p.proj_nr = 2
p.exp_cond = 'target'
p.roi = 'P2_LH_PPC'
p.hemi = 'lh'
p.mask_type = 'curv'
p.phi = 90.
p.radius = 65.
projects.update({p.proj_nr: {p.exp_cond: p}})

# P3
p = project_template()
p.proj_nr = 3
p.exp_cond = 'target'
p.roi = 'P4_LH_IFG'
p.hemi = 'lh'
p.mask_type = 'curv'
p.phi = 75.
p.radius = 50.
projects.update({p.proj_nr: {p.exp_cond: p}})

# P4
p = project_template()
p.proj_nr = 4
p.exp_cond = 'target'
p.roi = 'P4_LH_IFG'
p.hemi = 'lh'
p.mask_type = 'curv'
p.phi = 75.
p.radius = 50.
projects.update({p.proj_nr: {p.exp_cond: p}})

# P5
p = project_template()
p.proj_nr = 5
p.exp_cond = 'target'
p.roi = 'P5_LH_M1'
p.hemi = 'lh'
p.mask_type = 'curv'
p.phi = 90.
p.radius = 55.
projects.update({p.proj_nr: {p.exp_cond: p}})

# P6
p = project_template()
p.proj_nr = 6
p.exp_cond = 'target'
p.roi = 'P6_RH_CB.nii.gz'
p.hemi = 'cereb'
p.mask_type = 'mnivol'
p.phi = 90.
p.radius = 60.
projects.update({p.proj_nr: {p.exp_cond: p}})

# P7
p = project_template()
p.proj_nr = 7
p.exp_cond = 'target'
p.roi = 'P7_RH_DLPFC'
p.hemi = 'rh'
p.mask_type = 'curv'
p.phi = 30.
p.radius = 45.
projects.update({p.proj_nr: {p.exp_cond: p}})

# P8
p = project_template()
p.proj_nr = 8
p.exp_cond = 'target'
p.roi = 'P8_LH_DLPFC'
p.hemi = 'lh'
p.mask_type = 'curv'
p.phi = 75.
p.radius = 40.
projects.update({p.proj_nr: {p.exp_cond: p}})



# definition of CONTROL conditions

# P1
p = project_template()
p.proj_nr = 1
p.exp_cond = 'control'
p.roi = 'P1_LH_M1_control'
p.hemi = 'lh'
p.mask_type = 'curv'
p.phi = 90.
p.radius = 55.
projects[p.proj_nr].update({p.exp_cond: p})

# P2
p = project_template()
p.proj_nr = 2
p.exp_cond = 'control'
p.roi = 'P1_RH_OTC'
p.hemi = 'rh'
p.mask_type = 'curv'
p.phi = 35.
p.radius = 40.
projects[p.proj_nr].update({p.exp_cond: p})

# P3
p = project_template()
p.proj_nr = 3
p.exp_cond = 'control'
p.roi = 'P3_RH_M1_control'
p.hemi = 'rh'
p.mask_type = 'curv'
p.phi = 90.
p.radius = 55.
projects[p.proj_nr].update({p.exp_cond: p})

# P4
p = project_template()
p.proj_nr = 4
p.exp_cond = 'control'
p.roi = 'P1_RH_OTC'
p.hemi = 'rh'
p.mask_type = 'curv'
p.phi = 35.
p.radius = 40.
projects[p.proj_nr].update({p.exp_cond: p})

# P5
p = project_template()
p.proj_nr = 5
p.exp_cond = 'control'
p.roi = 'P4_LH_IFG'
p.hemi = 'lh'
p.mask_type = 'curv'
p.phi = 75.
p.radius = 50.
projects[p.proj_nr].update({p.exp_cond: p})

# P6
p = project_template()
p.proj_nr = 6
p.exp_cond = 'control'
p.roi = 'P4_LH_IFG'
p.hemi = 'lh'
p.mask_type = 'curv'
p.phi = 75.
p.radius = 50.
projects[p.proj_nr].update({p.exp_cond: p})

# P7
p = project_template()
p.proj_nr = 7
p.exp_cond = 'control'
p.roi = 'P1_LH_M1_control'
p.hemi = 'lh'
p.mask_type = 'curv'
p.phi = 90.
p.radius = 55.
projects[p.proj_nr].update({p.exp_cond: p})

# P8
p = project_template()
p.proj_nr = 8
p.exp_cond = 'control'
p.roi = 'P8_RH_M1_control'
p.hemi = 'rh'
p.mask_type = 'curv'
p.phi = 90.
p.radius = 55.
projects[p.proj_nr].update({p.exp_cond: p})