#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os.path import join as pjoin

import nibabel as nib
import numpy as np
import vtk
from fury import utils, window  # , actor

from dmriseg.anatomy.utils import Axis
from dmriseg.image.utils import (
    create_mask_from_label_image,
    extract_roi_from_label_image,
)
from dmriseg.visualization.actor_utils import (
    create_clip_actor,
    create_volume_slice_actor,
)
from dmriseg.visualization.scene_utils import (
    compose_scene,
    contour_actor_kwargs_name,
    create_slice_roi_scene,
)
from dmriseg.visualization.utils import AnatomicalView, compute_central_slices
from dmriseg.visualization.vtk_utils import (
    compute_mask_planar_intersection,
    extract_polydata_from_image_data,
    smooth_polydata,
)


def test_contour_from_roi():

    path = "/mnt/data/tractodata_testing_data/datasets/hcp_ya/100307/"
    ref_anat_img_fname = pjoin(path, "T1w_acpc_dc_restore_1.25.nii.gz")
    ref_anat_img = nib.load(ref_anat_img_fname)

    mask_fname = pjoin(
        path, "wmparc_brain_mask_reshape_T1w_acpc_dc_restore_1.25.nii.gz"
    )
    # brain_mask = nib.load(mask_fname)

    actors = []
    # contour_color = (1, 0, 0)
    # contour_opacity = 0.7

    # Extract the contour of the mask
    # brain_mask_actor = actor.contour_from_roi(
    #    brain_mask.get_fdata(),
    #    ref_anat_img.affine,
    #    contour_color,
    #    contour_opacity,
    # )
    # actors.append(brain_mask_actor)

    # ToDo
    # Experiment with actor.contour_from_label(data, affine=None, color=None)
    # It calls contour_from_roi, and I can add different RGA colors for each
    # label id in the data

    mask_reader = vtk.vtkNIFTIImageReader()
    mask_reader.SetFileName(mask_fname)
    mask_reader.Update()
    vtk_image_data = mask_reader.GetOutput()
    polydata = extract_polydata_from_image_data(vtk_image_data)

    smoothed_polydata = smooth_polydata(polydata, 50)
    surface_actor = utils.get_actor_from_polydata(smoothed_polydata)
    surface_actor_properties = surface_actor.GetProperty()
    surface_actor_properties.SetOpacity(1.0)
    surface_actor.SetProperty(surface_actor_properties)

    # ToDo
    # Set the normals pointing towards the outside to avoid seeing holes; maybe
    # in the smooth_polydata method
    actors.append(surface_actor)

    slices = compute_central_slices(ref_anat_img)
    axis = Axis.CORONAL
    anatomical_view = AnatomicalView.CORONAL_ANTERIOR

    slice_actor = create_volume_slice_actor(
        ref_anat_img.get_fdata(),
        axis,
        slices[1],
        value_range=None,
        opacity=0.5,
        lookup_colormap=None,
        interpolation="linear",
        picking_tol=0.025,
    )
    actors.append(slice_actor)

    # scene.set_camera(position=view_position, view_up=view_up_vector,
    #                 focal_point=focal_point)

    scene = compose_scene(
        actors,
        anatomical_view=anatomical_view,
        background=window.colors.black,
    )

    size = (900, 900)
    reset_camera = False
    showm = window.ShowManager(scene, size=size, reset_camera=reset_camera)
    showm.initialize()
    showm.start()

    tmp_path = "/home/jhlegarreta/Downloads"
    filename = pjoin(tmp_path, "test_contour_from_roi.png")
    window.snapshot(scene, fname=filename, size=size)


def test_compute_mask_planar_intersection():
    import scipy.ndimage as ndimage

    path = "/home/jhlegarreta/data/tractodata_testing_data/datasets/hcp_ya/100307/"
    ref_anat_img_fname = pjoin(path, "T1w_acpc_dc_restore_1.25.nii.gz")
    ref_anat_img = nib.load(ref_anat_img_fname)

    mask_fname = pjoin(
        path, "wmparc_brain_mask_reshape_T1w_acpc_dc_restore_1.25.nii.gz"
    )
    brain_mask = nib.load(mask_fname)

    normal = (0, 0, 1)
    origin = ndimage.measurements.center_of_mass(ref_anat_img.get_fdata())

    clip_plane = vtk.vtkPlane()
    clip_plane.SetNormal(normal)
    clip_plane.SetOrigin(origin)

    # Compute the intersection in the origin
    affine = np.eye(4)

    fill_contour = False

    # Compute the intersection
    polydata_algorithm = compute_mask_planar_intersection(
        brain_mask.get_fdata(),
        clip_plane,
        affine,
        fill_contour=fill_contour,
    )

    color = None
    opacity = 0.7
    linewidth = 4.5

    clip_actor = create_clip_actor(
        polydata_algorithm,
        color=color,
        opacity=opacity,
        linewidth=linewidth,
    )

    scene = window.Scene()
    scene.add(clip_actor)

    size = (900, 900)
    tmp_path = "/home/jhlegarreta/Downloads"
    filename = pjoin(tmp_path, "compute_mask_planar_intersection.png")
    window.snapshot(scene, fname=filename, size=size)


def test_create_slice_roi_scene():

    # Load a SUIT segmentation
    fname = "/mnt/data/connectome/suit/results/cer_seg_100307.nii"
    label_img = nib.load(fname)

    label8 = 8
    label10 = 10
    label = [label8, label10]

    roi_img = []
    # path = "/mnt/data/tractodata_testing_data/datasets/suit/"
    for _label in label:
        # Extract ROI using the label of interest
        _label_img = extract_roi_from_label_image(label_img, _label)

        # Create a binary image
        img_data = create_mask_from_label_image(_label_img)
        img = nib.nifti1.Nifti1Image(
            img_data.astype(np.uint8),
            affine=label_img.affine,
            header=label_img.header,
        )
        # ToDo
        # Not sure why the volume shows no data in 3D slicer
        # if _label == 8:
        #    out_fname = pjoin(path, "cer_seg_100307_label8.nii")
        #    img.header.set_data_dtype(np.uint8)
        #    nib.save(img, out_fname)
        roi_img.append(img)

    # Load the corresponding T1
    fname = "/mnt/data/connectome/s1200_download_on_20170516/downloaded/3T_structural_preproc/100307_3T_Structural_preproc/100307/T1w/T1w_acpc_dc_restore_brain.nii.gz"
    structural_img = nib.load(fname)

    # cmap = get_atlas_cmap(Atlas.DKT)
    color = [np.array([1, 0, 0]), np.array([0, 1, 0])]
    opacity = [1.0, 0.8]
    contour_actor_kwargs = dict({"color": color, "opacity": opacity})
    kwargs = dict({contour_actor_kwargs_name: contour_actor_kwargs})

    axis = Axis.AXIAL
    slice_idx = 66
    scene = create_slice_roi_scene(
        structural_img, roi_img, axis, slice_idx, **kwargs
    )

    size = (1024, 720)
    # reset_camera = False
    # showm = window.ShowManager(scene, size=size, reset_camera=reset_camera)
    # showm.initialize()
    # showm.start()

    tmp_path = "/home/jhlegarreta/Downloads"
    filename = pjoin(tmp_path, "test_create_slice_roi_scene.png")
    window.snapshot(scene, fname=filename, size=size)


if __name__ == "__main__":
    # test_contour_from_roi()
    # test_compute_mask_planar_intersection()
    test_create_slice_roi_scene()
