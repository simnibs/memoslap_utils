import os
from os.path import join

import pyvista as pv
import numpy as np
import re
import matplotlib.pyplot as plt
import pickle
import nibabel as nib

from simnibs.utils.file_finder import SubjectFiles


def _pvmesh_from_skin_geo(infile):
    # make pyvista mesh out of a skin geo file
    with open(infile, "rt") as f:
        lines = f.readlines()
    points = []
    faces = []
    for idx, line in enumerate(lines[1:]):
        mch = re.match("ST\((.*), (.*), (.*),(.*), (.*), (.*),(.*), (.*), (.*)\)",
                       line)
        if mch:
            these_points = np.array([float(f) for f in mch.groups(0)]).reshape(-1, 3)
            this_face = np.array([0, 1, 2]) + idx*3
            points.append(these_points)
            faces.append(np.hstack((3, this_face)))
    points = np.vstack(points)
    faces = np.hstack(faces)
    mesh = pv.PolyData(points, faces)
    return mesh


# def _elpos_from_geo(infile):
#     # get electrode points from geo file
#     with open(infile, "rt") as f:
#         lines = f.readlines()
#     points = []
#     for idx, line in enumerate(lines[1:]):
#         mch = re.match("SP\((.*), (.*), (.*)\){(.*)}", line)
#         if mch:
#             these_points = np.array([float(f) for f in mch.groups(0)])
#             points.append(these_points)
#     points = np.vstack(points)

#     return points


# def _elpos_from_brainsight(infile):
#     # get electrodes coords out of a brainsight file
#     with open(infile, "rt") as f:
#         lines = f.readlines()
#     coords = {}
#     for idx, line in enumerate(lines[1:]):
#         mch = re.match("(.*_electrode(_\d)?)\t([\d.-]*)\t([\d.-]*)\t([\d.-]*)", line)
#         if mch:
#             coords[mch.groups()[0]] = np.array([float(c) for c in mch.groups()[2:5]])
#     return coords


def _elec_plot_geo(mesh, elecs, cam_dist=200., return_foc=False, elec_rad=5.,
                   background="white"):
    # plot electrodes on a scalp. mesh produced by pv_mesh_from_skin_geo,
    # elecs from _elpos_from_geo

    # renderer
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background(background)
    # skin
    plotter.add_mesh(mesh, color=[.7, .5, .5])

    # electrodes
    surr_count = 0
    surr_cols = ["red", "green", "blue"]
    for elec in elecs:
        sph_mesh = pv.Sphere(elec_rad, elec[:3])
        if int(elec[-1]) == 1:
            color = "black"
            foc = elec[:3]
        else:
            color = surr_cols[surr_count]
            surr_count += 1
        plotter.add_mesh(sph_mesh, color=color)

    # camera work
    # centre point of this electrode for focal point
    plotter.camera.focal_point = foc
    # normed vector from origin to focal point
    norm_vec = foc / np.linalg.norm(foc)
    pos = foc + norm_vec * cam_dist
    plotter.camera.position = pos

    image = plotter.screenshot(None, return_img=True)
    plotter.close()
    if return_foc:
        return image, foc
    else:
        return image


def _mag_plot(mesh, foc="elec", pos=None, cam_dist=200., clim=[0., .3],
              return_foc=False, cmap="jet", background="white"):
    # plot field magnitude from the results mesh file
    # renderer
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background(background)
    # get gm
    if "magnE" in mesh.array_names:
        gm_inds = np.where(mesh.cell_data["gmsh:physical"]==2)[0]
        gm_cells = mesh.extract_cells(gm_inds)
        plotter.add_mesh(gm_cells, scalars="magnE", cmap=cmap,
                         clim=clim, scalar_bar_args={"color":"black"})
    else:
        plotter.add_mesh(mesh, scalars="E_magn", cmap=cmap,
                         clim=clim, scalar_bar_args={"color":"black"})

    # camera work
    if np.any(foc == "elec"):
        centre_inds = np.where(mesh.cell_data["gmsh:physical"]==101)[0]
        centre_cells = mesh.extract_cells(centre_inds)
        foc = centre_cells.cell_centers().points.mean(axis=0)

    # centre point of this electrode for focal point
    plotter.camera.focal_point = foc
    # normed vector from origin to focal point
    norm_vec = foc / np.linalg.norm(foc)
    if pos is None:
        pos = foc + norm_vec * cam_dist
    plotter.camera.position = pos

    image = plotter.screenshot(None, return_img=True)
    plotter.close()
    if return_foc:
        return image, foc
    else:
        return image

def _roi_plot(mesh, foc="centre", cam_dist=200., clim=[0., .6], cmap=["blue", "white"], background="white"):
    # plot the roi from the results mesh file
    # renderer
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background(background)
    plotter.add_mesh(mesh, show_scalar_bar=False, scalars="mask", cmap=cmap)

    # camera work
    if np.any(foc == "centre"):
        arr_name = mesh.array_names[0] # assumes first is ROI array
        roi_inds = np.where(mesh[arr_name]==1)[0]
        roi_points = mesh.extract_points(roi_inds)
        foc = roi_points.points.mean(axis=0)
    elif isinstance(foc, (list, tuple, np.ndarray)):
        foc = foc
    else:
        raise ValueError("Unrecognised 'foc' value")

    # centre point of this electrode for focal point
    plotter.camera.focal_point = foc
    # normed vector from origin to focal point
    norm_vec = foc / np.linalg.norm(foc)
    pos = foc + norm_vec * cam_dist
    plotter.camera.position = pos

    image = plotter.screenshot(None, return_img=True)
    plotter.close()
    return image


def _get_RAS2nexstim(img):
    # this might be the internal Nexstim coordinate system in [mm]
    # min: RIP (0,0,0)
    # max: LSA ( (nvox_RL-1)*vox_size_RL,
    #            (nvox_IS-1)* vox_size_IS,
    #            (nvox_PA-1)*vox_size_PA )
    #
    # x: Right to Left - Sagittal
    # y: Inferior to Superior - Axial
    # z: Posterior to Anterior - Coronal
    Q=img.get_qform()
    
    # get trafo from image voxel space to internal voxel space of nexstim
    axcodes = nib.orientations.aff2axcodes(Q)
    ornt = nib.orientations.axcodes2ornt(axcodes, (('R','L'),('I','S'),('P','A')))
    M=nib.orientations.inv_ornt_aff(ornt,img.shape)
    
    # scaling matrix
    voxsize = np.linalg.norm(Q[:3,:3],axis=0)
    M2 = np.zeros((4,4))
    M2[0,0]=voxsize[0]
    M2[1,1]=voxsize[1]
    M2[2,2]=voxsize[2]
    M2[3,3]=1.
    
    # trafo from "real world" space of nexstim to image voxel space
    img2vox=np.matmul(M,np.linalg.inv(M2))
    vox2img=np.linalg.inv(img2vox)
    
    RAS2nexstim = np.matmul(vox2img,np.linalg.inv(Q))
    
    return RAS2nexstim


def _convert_pts_to_nexstim(pts, subject_path):
    subject_files = SubjectFiles(subpath=subject_path)
    
    img = nib.load(subject_files.reference_volume)
    RAS2nexstim = _get_RAS2nexstim(img)
    
    pts_nexstim = np.zeros_like(pts)
    pts_nexstim[:,3] = pts[:,3]
    for i in range(len(pts_nexstim)):        
        pts_nexstim[i,:3] = np.matmul(RAS2nexstim,
                                      np.hstack((pts[i,:3], 1.))
                                      )[:3]
    return pts_nexstim


def placement_guide(mesh_path, radius, out_file = None, cam_dist=500.,
                    nexstim=False, subject_path=None):

    mesh_path = os.path.abspath(mesh_path)
    basename = os.path.basename(mesh_path)

    # get info out of results object
    with open(join(mesh_path, 'simnibs_memoslap_results.pkl'), "rb") as f:
        results = pickle.load(f)
    [project,
     pos_center,
     pos_surround,
     res_list,
     res_list_raw,
     analysis,
     version,
     date
    ] = results

    rad_idx = analysis["radius"].index(int(radius))
    radius = analysis["radius"][rad_idx] # prevents errors when using radius as key

    proj_nr = project["proj_nr"]

    points = np.vstack((pos_center,pos_surround[radius]))
    points = np.hstack((points,np.reshape([1,0,0,0],(4,1))))

    skin_file = basename+'_skin.geo'
    subj = basename.split('_',2)[-1]
    if out_file is None:
        if nexstim:
            out_file = basename+'_'+str(int(radius))+'_placement_NEXSTIM.pdf'
        else:
            out_file = basename+'_'+str(int(radius))+'_placement.pdf'

    # load mesh
    mesh = _pvmesh_from_skin_geo(join(mesh_path, skin_file))

    # text part of report
    colors = ["black", "red", "green", "blue"]
    y_base = 0.99
    x_base = 0.01
    fig, axes = plt.subplots(2, 1, figsize=(15.32, 21.6))
    kwargs = {"transform":axes[0].transAxes, "verticalalignment":"top"}
    axes[0].text(x_base, y_base, "MeMoSLAP Placement Guide\n",
                 weight="bold", fontsize=38, **kwargs)
    axes[0].text(x_base, y_base-0.1,
                 f"Project P{proj_nr}\nID: {subj}\n"
                 f"Radius: {int(radius)}mm",
                 fontsize=28, **kwargs)
    axes[0].text(x_base, y_base-0.35, "Coordinates (XYZ)", fontsize=28,
                 weight="bold", **kwargs,)
    if nexstim:
        axes[0].text(x_base, y_base-0.45, "NEXSTIM", fontsize=28, 
                     color="red", weight="bold", **kwargs)
        coords_printed = _convert_pts_to_nexstim(points, subject_path)
    else:
        axes[0].text(x_base, y_base-0.45, "RAS", fontsize=28, **kwargs)
        coords_printed = points
        
    # for idx, ((k, v), color) in enumerate(zip(coords.items(), colors)):
    #     k = k.replace("_electrode", "")
    #     k = k.replace("surround", "rad")
    #     v = np.round(v, 2)
    #     axes[0].text(x_base, y_base-0.55-idx*0.1, f"{k}: {v[0]:.2f} "
    #                  f"{v[1]}, {v[2]}", fontsize=28, color=color, **kwargs)
    for idx in range(4):
        v = np.round(coords_printed[idx,:3], 2)
        if idx == 0:
            k = 'center'
        else:
            k = 'pad '+str(idx)
        axes[0].text(x_base, y_base-0.55-idx*0.1, f"{k}: {v[0]:.2f} "
                      f"{v[1]}, {v[2]}", fontsize=28, color=colors[idx], **kwargs)
    axes[0].axis("off")

    # image
    elec_img = _elec_plot_geo(mesh, points, return_foc=False,
                             cam_dist=cam_dist)
    axes[1].imshow(elec_img)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(join(mesh_path, out_file))
    plt.close("all")


def internal_report(mesh_path, radius, out_file=None, cam_dist=500.,
                    cond_order="Unspecified"):

    mesh_path = os.path.abspath(mesh_path)
    basename = os.path.basename(mesh_path)

    # get info out of results object
    with open(join(mesh_path, 'simnibs_memoslap_results.pkl'), "rb") as f:
        results = pickle.load(f)
    [project,
     pos_center,
     pos_surround,
     res_list,
     res_list_raw,
     analysis,
     version,
     date
    ] = results

    rad_idx = analysis["radius"].index(int(radius))
    radius = analysis["radius"][rad_idx] # prevents errors when using radius as key

    proj_nr = project["proj_nr"]
    phi = project["phi"]
    alloc = project["exp_cond"]

    points = np.vstack((pos_center,pos_surround[radius]))
    points = np.hstack((points,np.reshape([1,0,0,0],(4,1))))
    mesh_file = os.path.split(res_list[radius])[1]

    fieldmed = analysis["roi_median"]["E_magn"][rad_idx]
    fieldsumsq = analysis["roi_squared"]["E_magn"][rad_idx]
    foc = analysis["focality"]["E_magn"][rad_idx]

    skin_file = basename+'_skin.geo'
    subj = basename.split('_',2)[-1]
    if out_file is None:
        out_file = basename+'_'+str(int(radius))+'_internal.pdf'

    mos_str = """
              AB
              CD
              """
    fig, axes = plt.subplot_mosaic(mos_str, figsize=(15.32, 21.6))
    # report text
    y_base = 0.99
    x_base = 0.01
    kwargs = {"transform":axes["A"].transAxes, "verticalalignment":"top"}
    axes["A"].text(x_base, y_base, "MeMoSLAP FEM Summary",
                 **kwargs, weight="bold", fontsize=38)
    axes["A"].text(x_base, y_base-0.1,
                 f"Project P{proj_nr}\n"
                 f"ID: {subj}\n"
                 f"Radius: {int(radius)}mm\n"
                 f"Phi: {int(phi)}\n\n"
                 f"Group Allocation: {alloc}\n"
                 f"Condition order: {cond_order}\n\n"
                 f"Median field magnitude: {np.round(fieldmed, 3)}\n"
                 f"Field magnitude sum²: {int(np.round(fieldsumsq))}\n"
                 f"Focality: {int(np.round(foc))}\n\n"
                 f"Simulated with version {version}\n"
                 f"  on {date.strftime('%d.%m.%Y %H:%M')}",
                 **kwargs, fontsize=28)

    axes["A"].axis("off")

    # electrodes
    mesh = _pvmesh_from_skin_geo(join(mesh_path, skin_file))
    elec_img, foc = _elec_plot_geo(mesh, points, return_foc=True,
                                  cam_dist=cam_dist)
    
    axes["B"].imshow(elec_img)
    axes["B"].set_title("Electrodes", fontsize=30, weight="bold")
    axes["B"].axis("off")

    # bottom panel
    mesh = pv.read(join(mesh_path, mesh_file))
    mag_img = _mag_plot(mesh, foc=foc, cam_dist=cam_dist * .7)
    axes["C"].imshow(mag_img)
    axes["C"].set_title("Field Magnitude", fontsize=30, weight="bold")
    axes["C"].axis("off")

    roi_img = _roi_plot(mesh, foc=foc, cam_dist=cam_dist * .7)
    axes["D"].imshow(roi_img)
    axes["D"].set_title("ROI", fontsize=30, weight="bold")
    axes["D"].axis("off")

    plt.savefig(join(mesh_path, out_file))
    plt.close("all")
