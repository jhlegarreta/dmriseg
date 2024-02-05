#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os.path import join as pjoin

import nibabel as nib
import numpy as np
import vtk
from fury import window

from dmriseg.visualization.actor_utils import create_clip_actor
from dmriseg.visualization.vtk_utils import compute_mask_planar_intersection


def test_plot_contour_variability():
    import scipy.ndimage as ndimage

    # ToDo
    # Data should be in a common space

    path = "/mnt/data/test_data/tractodata/datasets/hcp_ya/100307/"
    ref_anat_img_fname = pjoin(path, "T1w_acpc_dc_restore_1.25.nii.gz")
    ref_anat_img = nib.load(ref_anat_img_fname)

    mask_fname = pjoin(
        path, "wmparc_brain_mask_reshape_T1w_acpc_dc_restore_1.25.nii.gz"
    )
    brain_mask1 = nib.load(mask_fname)

    path = "/mnt/data/test_data/tractodata/datasets/hcp_ya/100408/"
    mask_fname = pjoin(
        path, "wmparc_brain_mask_reshape_T1w_acpc_dc_restore_1.25.nii.gz"
    )
    brain_mask2 = nib.load(mask_fname)

    normal = (0, 0, 1)
    origin = ndimage.measurements.center_of_mass(ref_anat_img.get_fdata())

    clip_plane = vtk.vtkPlane()
    clip_plane.SetNormal(normal)
    clip_plane.SetOrigin(origin)

    # Compute the intersection in the origin
    affine = np.eye(4)

    fill_contour = False

    # Erode the mask so that we get a different contour. Note that adding
    # noise to a mask does not produce the desired effect
    import scipy

    iterations = [1, 3]
    eroded_mask_img = []
    brain_mask = [brain_mask1, brain_mask2]
    for _iter, mask_img in zip(iterations, brain_mask):
        mask_data = mask_img.get_fdata()
        eroded_data = scipy.ndimage.binary_erosion(
            mask_data, iterations=_iter
        ).astype("uint8")
        eroded_mask = nib.nifti1.Nifti1Image(
            eroded_data, affine=mask_img.affine
        )
        eroded_mask_img.append(eroded_mask)

    clip_actor = []
    color = [tuple([1, 0, 0]), tuple([0, 1, 0])]
    opacity = 0.7
    linewidth = 4.5
    for _eroded_mask, _color in zip(eroded_mask_img, color):
        # Compute the intersection
        polydata_algorithm = compute_mask_planar_intersection(
            _eroded_mask.get_fdata(),
            clip_plane,
            affine,
            fill_contour=fill_contour,
        )

        _actor = create_clip_actor(
            polydata_algorithm,
            color=_color,
            opacity=opacity,
            linewidth=linewidth,
        )
        clip_actor.append(_actor)

    scene = window.Scene()
    background = window.colors.white
    scene.background(background)

    for _actor in clip_actor:
        scene.add(_actor)

    size = (900, 900)
    tmp_path = "/home/jhlegarreta/Downloads"
    filename = pjoin(tmp_path, "plot_contour_variability_brainmask.png")
    window.snapshot(scene, fname=filename, size=size)


def test_plot_contour_variability2():
    import scipy.ndimage as ndimage

    # ToDo
    # Data should be in a common space

    path = "/mnt/data/test_data/tractodata/datasets/hcp_ya/100307"
    ref_anat_img_fname = pjoin(path, "T1w_acpc_dc_restore_1.25.nii.gz")
    ref_anat_img = nib.load(ref_anat_img_fname)

    path = "/mnt/data/test_data/tractodata/datasets/suit/"
    cereb_seg_fname1 = pjoin(path, "cer_seg_100307.nii")
    label_img1 = nib.load(cereb_seg_fname1)

    path = "/mnt/data/test_data/tractodata/datasets/suit/"
    cereb_seg_fname2 = pjoin(path, "cer_seg_100408.nii")
    label_img2 = nib.load(cereb_seg_fname2)

    # Extract ROI using the label of interest
    from dmriseg.image.utils import (
        create_mask_from_label_image,
        extract_roi_from_label_image,
    )

    label = 8
    _label_img1 = extract_roi_from_label_image(label_img1, label)
    _label_img2 = extract_roi_from_label_image(label_img2, label)

    # Create a binary image
    img_data1 = create_mask_from_label_image(_label_img1)
    roi_img1 = nib.nifti1.Nifti1Image(img_data1, affine=label_img1.affine)
    img_data2 = create_mask_from_label_image(_label_img2)
    roi_img2 = nib.nifti1.Nifti1Image(img_data2, affine=label_img2.affine)

    normal = (0, 0, 1)
    origin = ndimage.measurements.center_of_mass(ref_anat_img.get_fdata())

    clip_plane = vtk.vtkPlane()
    clip_plane.SetNormal(normal)
    clip_plane.SetOrigin(origin)

    # Compute the intersection in the origin
    affine = np.eye(4)

    fill_contour = False
    clip_actor = []
    # ToDo
    # Use an R, matplotlib, seaborn, bokeh colormap cycle; or else use a black
    # brush with varying opacity as in the Diedrichsen 2011 NIMG Fig3 paper
    color = [tuple([1, 0, 0]), tuple([0, 1, 0])]
    opacity = 0.7
    linewidth = 4.5
    roi_img = [roi_img1, roi_img2]
    for _roi_img, _color in zip(roi_img, color):
        # Compute the intersection
        polydata_algorithm = compute_mask_planar_intersection(
            _roi_img.get_fdata(),
            clip_plane,
            affine,
            fill_contour=fill_contour,
        )

        _actor = create_clip_actor(
            polydata_algorithm,
            color=_color,
            opacity=opacity,
            linewidth=linewidth,
        )
        clip_actor.append(_actor)

    scene = window.Scene()
    background = window.colors.white
    scene.background(background)

    for _actor in clip_actor:
        scene.add(_actor)

    size = (900, 900)
    tmp_path = "/home/jhlegarreta/Downloads"
    filename = pjoin(tmp_path, "plot_contour_variability_cer_seg_label8.png")
    window.snapshot(scene, fname=filename, size=size)


if __name__ == "__main__":
    test_plot_contour_variability()
    test_plot_contour_variability2()
