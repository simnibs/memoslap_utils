# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 20:23:41 2023

@author: axthi
"""

import numpy as np
import os
from copy import deepcopy
from scipy.spatial.transform import Rotation as R

from simnibs import mesh_io, sim_struct
from simnibs.utils.file_finder import SubjectFiles
from simnibs.utils.simnibs_logger import logger


def sphereFit(pts, bounds = None):
    """
    Fit a circle or sphere to a point cloud.

    returns the radius and center points of the best fit sphere
    (adapted from https://jekel.me/2015/Least-Squares-Sphere-Fit/)

    Parameters
    ----------
    pts : array (Nx2) or (Nx3)
        point cloud

    Returns
    -------
    R : float64
        radius
    centre: ndarray (2,) or (3,)
        centre position

    """
    A = np.hstack((2*pts, np.ones((pts.shape[0],1)) ))
    f = np.sum(pts**2,1)
    C, residuals, rank, singval = np.linalg.lstsq(A,f,rcond=None)

    dim = pts.shape[1]
    R = np.sqrt(np.sum(C[0:dim]**2) + C[dim])
    return R, C[0:dim]


def sph2cart(az, el, r): # phi, theta, radius
    """Conversion from spherical to cartesian coordinates."""
    rcos_theta = r * np.cos(el)
    pts = np.zeros(( (3,) + rcos_theta.shape ))
    pts[0,:] = rcos_theta * np.cos(az)
    pts[1,:] = rcos_theta * np.sin(az)
    pts[2,:] = r * np.sin(el)
    return pts


def show_for_debugging(m,sph_centre,radius,P_centre,P_surround,surround_fit,M_sph):
    """Show some results in gmsh for debugging."""
    import tempfile
    fn_geo=tempfile.NamedTemporaryFile(suffix='.geo').name
    mesh_io.write_geo_spheres(sph_centre.reshape((1,3)),
                              fn_geo, name = 'center', mode = 'bw')
    mesh_io.write_geo_spheres(( M_sph @ [radius,0,0,1] )[:3].reshape((1,3)),
                              fn_geo, name = 'x', mode = 'ba')
    mesh_io.write_geo_spheres(( M_sph @ [0,radius,0,1] )[:3].reshape((1,3)),
                              fn_geo, name = 'y', mode = 'ba')
    mesh_io.write_geo_spheres(( M_sph @ [0,0,radius,1] )[:3].reshape((1,3)),
                              fn_geo, name = 'z', mode = 'ba')

    TH_DEBUG = np.arange(-1.0,1.01,0.1)*np.pi
    PHI_DEBUG = np.arange(0.,1.01,0.05)*2*np.pi
    TH_DEBUG, PHI_DEBUG = np.meshgrid(TH_DEBUG, PHI_DEBUG)
    TH_DEBUG = TH_DEBUG.flatten()
    PHI_DEBUG = PHI_DEBUG.flatten()
    R_DEBUG = radius*np.ones_like(TH_DEBUG)

    pts=sph2cart(PHI_DEBUG, TH_DEBUG, R_DEBUG)
    pts = np.vstack(( pts, np.ones((1,pts.shape[1])) ))
    mesh_io.write_geo_spheres((M_sph @ pts)[:3,:].T, fn_geo,
                              name = 'sphere', mode = 'ba')

    mesh_io.write_geo_spheres( P_centre.reshape((1,3)),
                               fn_geo, name = 'centre', mode = 'ba')
    for i in range(len(P_surround)):
        mesh_io.write_geo_spheres( P_surround[i].reshape((1,3)),
                                  fn_geo, name = 'surr '+str(i), mode = 'ba')

    N_pts = 50
    for i in range(len(surround_fit)):
        tmp_centre = surround_fit[i][0]
        tmp_r = surround_fit[i][1]
        tmp_theta = surround_fit[i][2]
        tmp_theta_z0 = surround_fit[i][3]
        tmp_M = surround_fit[i][4]

        tmp_arc = np.vstack((
            tmp_r*np.sin(tmp_theta_z0 + (tmp_theta-tmp_theta_z0)*np.arange(N_pts)/(N_pts-1)) + tmp_centre[0],
            np.zeros((1,N_pts)),
            tmp_r*np.cos(tmp_theta_z0 + (tmp_theta-tmp_theta_z0)*np.arange(N_pts)/(N_pts-1)) + tmp_centre[1],
            np.ones((1,N_pts))
            ))
        tmp_arc=(tmp_M @ tmp_arc)[:3].T
        mesh_io.write_geo_spheres(tmp_arc, fn_geo, name = 'arc '+str(i), mode = 'ba')

    vis = mesh_io.gmsh_view.Visualization(m)
    vis.add_merge(fn_geo)
    vis.show()
    os.remove(fn_geo)


def get_surround_pos(center_pos, fnamehead, radius_surround=50, N=4,
                     pos_dir_1stsurround=None, phis_surround=None,
                     tissue_idx=1005, DEBUG=False):
    """
    Determine the positions of surround electrodes.

    Parameters
    ----------
    center_pos : array (3,) or string
        Center position of the central electrode.
    fnamehead : string
        Filename of head mesh.
    radius_surround : float or array (N,), optional
        Distance (centre-to-centre) between the centre and surround
        electrodes. The default is 50. Either a single number (same
        radius for all electrodes) or an array with N numbers
        (N: number of surround electrodes)
    N : int, optional
        Number of surround electrodes. The default is 4.
    pos_dir_1stsurround : array (3,) or string, optional
        A position indicating the direction from center_pos to
        the position of the first surround electrode. The default is None.
    phis_surround : array (N,), optional
        Angles in degree at which the electrodes will be place relative to the
        direction defined by pos_dir_1stsurround. The default is None, in which
        case the electrodes will be placed at [0, 1/N*360, ..., (N-1)/N*360]
        degrees.
    tissue_idx : int, optional
        Index of the tissue on which the surround positions will be planned on.
        (standard: 1005 for skin)
    DEBUG : Boolean, optional
        When set to True, a visualization in gmsh will open for control
        (standard: False)

    Returns
    -------
    P_surround : list of arrays (3,)
        List of the centre positions of the surround electrodes.

    """

    # replace electrode name with position if needed
    # and get skin ROI around centre position
    ff = SubjectFiles(fnamehead=fnamehead)
    tmp = sim_struct.ELECTRODE()
    tmp.centre = center_pos
    tmp.substitute_positions_from_cap(ff.get_eeg_cap())

    m = mesh_io.read_msh(fnamehead)
    if tissue_idx < 1000:
        tissue_idx += 1000
    idx = (m.elm.elm_type == 2) & (
        (m.elm.tag1 == tissue_idx) | (m.elm.tag1 == tissue_idx-1000))
    m = m.crop_mesh(elements=m.elm.elm_number[idx])
    P_centre = m.find_closest_element(tmp.centre)
    idx = np.sum((m.nodes[:] - P_centre)**2,
                 1) <= (np.max(radius_surround)+10)**2
    m = m.crop_mesh(nodes=m.nodes.node_number[idx])
    idx = m.elm.connected_components()
    m = m.crop_mesh(elements=max(idx, key=np.size))

    # fit sphere to skin ROI to build local coordinate system
    #   origin: sphere center
    #   x-axis: direction of first surround
    #   z-axis: from sphere center to centre electrode
    r_sph, sph_centre = sphereFit(m.nodes[:])

    M_sph = np.eye(4)
    M_sph[:3, 3] = sph_centre
    tmp = P_centre - sph_centre
    M_sph[:3, 2] = tmp/np.linalg.norm(tmp)
    # direction of first surround
    if pos_dir_1stsurround is not None:
        # replace electrode name with position if needed
        tmp = sim_struct.ELECTRODE()
        tmp.centre = pos_dir_1stsurround
        tmp.substitute_positions_from_cap(ff.get_eeg_cap())
        tmp = tmp.centre - P_centre  # this is not orthogonal to Z
    else:
        # get a vector orthogonal to z-axis
        tmp = np.cross(M_sph[:3, 2], np.eye(3))
        tmp = tmp[:, np.argmax(np.linalg.norm(tmp, axis=1))]
    M_sph[:3, 1] = np.cross(M_sph[:3, 2], tmp)
    M_sph[:3, 1] /= np.linalg.norm(M_sph[:3, 1])
    M_sph[:3, 0] = np.cross(M_sph[:3, 1], M_sph[:3, 2])

    # fit arcs to the skin to get the distances accurate
    if phis_surround is not None:
        if len(phis_surround) != N:
            raise ValueError('exactly N angles are required')
        phis_surround = np.asarray(phis_surround)/180*np.pi  # convert to rad
    else:
        phis_surround = np.arange(N)/N*2*np.pi

    radius_surround = np.array(radius_surround)
    if radius_surround.size == 1:
        radius_surround = np.tile(radius_surround, N)

    N_pts = 50
    P_surround = []
    surround_fit = []
    for phi in range(N):
        theta_on_sph = radius_surround[phi]/r_sph
        arc = np.vstack((r_sph*np.sin(theta_on_sph*np.arange(N_pts)/(N_pts-1)),
                         np.zeros((1, N_pts)),
                         r_sph*np.cos(theta_on_sph *
                                      np.arange(N_pts)/(N_pts-1)),
                         np.ones((1, N_pts))))

        M_rot = np.eye(4)
        M_rot[:3, :3] = R.from_euler('z', phis_surround[phi]).as_dcm()
        M_to_world = M_sph @ M_rot
        M_from_world = np.linalg.inv(M_to_world)

        # project skin points into XZ-plane that contains the arc
        Pts = (M_to_world @ arc).T
        Pts[:, :3] = m.find_closest_element(Pts[:, :3])
        Pts = M_from_world @ Pts.T

        # fit individual arc
        r_arc, arc_centre = sphereFit(Pts[(0, 2), :].T)

        if np.abs(arc_centre[0]) > r_arc:
            # z-axis does not intersect with circle
            # --> use initial sphere instead
            r_arc = r_sph
            arc_centre *= 0

        theta_z0_on_arc = -np.arcsin(arc_centre[0]/r_arc)
        if arc_centre[1] < np.mean(Pts[2, :]):
            theta_on_arc = radius_surround[phi]/r_arc + theta_z0_on_arc
        else:
            # best fitting arc has opposite curvature compared
            # to initial sphere
            theta_z0_on_arc = - theta_z0_on_arc + np.pi
            theta_on_arc = theta_z0_on_arc - radius_surround[phi]/r_arc

        # get centre of surround electrode
        tmp = np.array((r_arc*np.sin(theta_on_arc) + arc_centre[0],
                        0.,
                        r_arc*np.cos(theta_on_arc) + arc_centre[1],
                        1.))
        P_surround.append(m.find_closest_element((M_to_world @ tmp).T[:3]))

        if DEBUG:
            surround_fit.append(
                (arc_centre, r_arc, theta_on_arc, theta_z0_on_arc, M_to_world))
    if DEBUG:
        print('achieved distances:')
        print(np.linalg.norm(np.array(P_surround)-P_centre, axis=1))
        # _show_for_debugging(m, sph_centre, r_sph, P_centre,
        #                     P_surround, surround_fit, M_sph)

    return P_surround


def expand_to_center_surround(S, subpath, radius_surround = 50, N = 4,
                              pos_dir_1stsurround = None, multichannel = False,
                              phis_surround=None, el_surround=None):
    """
    Generate a center-surround montage (standard: 4x1) from a TDCSLIST.

    The TDCSLIST has to contain only the center electrode. Copies of this
    electrode are then placed in a circle around the centre

    Parameters
    ----------
    S : TDCSLIST
        TDCSLIST with the center electrode.
    subpath : string
        m2m_folder of the subject
    radius_surround : float, optional
        Distance (centre-to-centre) between the centre and surround
        electrodes. The default is 50.
    N : int, optional
        Number of surround electrodes. The default is 4.
    pos_dir_1stsurround : array (3,) or string, optional
        A position indicating the direction from center_pos to
        the position of the first surround electrode. The default is None.
    multichannel : Boolean, optional
        When set to True, a multichannel stimulator with each suround channel
        receiving 1/N-th of the of the center channel will be simulated
        (standard: False, i.e. all surround electrodes connected to the
         same return channel).

    Returns
    -------
    S : TDCSLIST
        TDCSLIST with the surround electrodes added.

    """
    if S.type != 'TDCSLIST':
        raise TypeError('The first parameter needs to be a TDCSLIST.')
    if len(S.electrode) != 1:
        raise ValueError('The TDCSLIST has to contain exactly one ELECTRODE.')
    if not os.path.isdir(subpath):
        raise IOError('Could not find m2m-folder: {0}'.format(subpath))

    C = S.electrode[0]
    C.channelnr = 1  # Connect center to channel 1
    if not len(C.name):
        C.name = 'centre'

    # set surround channels and current strengths
    if type(S.currents) == float:
        C_current = S.currents
    else:
        C_current = S.currents[0]

    if multichannel:
        S.currents = -C_current/N*np.ones(N+1)
        S.currents[0] = C_current
        Channel_surround = np.arange(2,N+2)
    else:
        S.currents = [C_current, -C_current]
        Channel_surround = 2*np.ones(N,dtype = int)

    # get centers of surround electrodes
    ff = SubjectFiles(subpath=subpath)
    P_surround = get_surround_pos(C.centre, ff.fnamehead, radius_surround = radius_surround,
                                  N = N, pos_dir_1stsurround = pos_dir_1stsurround,
                                  phis_surround=phis_surround)

    if el_surround is not None:
        C = el_surround

    # get direction vector
    ydir = []
    if len(C.pos_ydir):
        tmp = deepcopy(C)
        tmp.substitute_positions_from_cap(ff.get_eeg_cap())
        ydir = tmp.pos_ydir - tmp.centre

    # add surround electrodes to TDCSLIST
    for i in range(N):
        S.electrode.append(deepcopy(C))
        El = S.electrode[-1]
        El.centre = P_surround[i]
        El.channelnr = Channel_surround[i]
        El.name = 'surround '+str(i+1)
        if len(ydir):
            El.pos_ydir = El.centre + ydir
    return S
