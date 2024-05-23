#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot the contour of the region of interest corresponding to the given label
value in a labelmap. If multiple labelmaps are given, they are colored according
to the given colormap name. Similar to the Diedrichsen 2011 NIMG Fig3 paper.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.ndimage as ndimage
import vtk
from fury import window
from matplotlib import pyplot as plt

from dmriseg.image.utils import (
    create_mask_from_label_image,
    extract_roi_from_label_image,
)
from dmriseg.visualization.actor_utils import create_clip_actor
from dmriseg.visualization.vtk_utils import compute_mask_planar_intersection


def plot_contour(
    ref_anat_img, label_imgs, label, colors, opacity=0.7, linewidth=4.5
):

    # Extract ROI using the label of interest
    roi_img = [extract_roi_from_label_image(img, label) for img in label_imgs]

    # Binarize ROIs
    mask_img_data = [create_mask_from_label_image(img) for img in roi_img]
    mask_img = [
        nib.nifti1.Nifti1Image(img_data, affine=img.affine)
        for img_data, img in zip(mask_img_data, label_imgs)
    ]

    normal = (0, 0, 1)
    origin = ndimage.center_of_mass(ref_anat_img.get_fdata())

    clip_plane = vtk.vtkPlane()
    clip_plane.SetNormal(normal)
    clip_plane.SetOrigin(origin)

    # Compute the intersection in the origin
    affine = np.eye(4)

    fill_contour = False
    clip_actor = []
    for _roi_img, _color in zip(mask_img, colors):
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

    return scene


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "in_ref_anat_img_fname",
        help="Reference anatomical image filename (*.nii.gz)",
        type=Path,
    )
    parser.add_argument(
        "out_fname",
        help="Output filename (*.png)",
        type=Path,
    )
    parser.add_argument(
        "label",
        help="Label of interest",
        type=int,
    )
    parser.add_argument(
        "--in_labelmap_fnames",
        help="Labelmap image filename (*.nii.gz)",
        nargs="+",
        type=Path,
    )
    parser.add_argument(
        "--cmap_name", help="Colormap name", type=str, default="Greys"
    )
    return parser


def _parse_args(parser):

    args = parser.parse_args()

    return args


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    # Data is assumed to be in a common space
    ref_anat_img = nib.load(args.in_ref_anat_img_fname)
    label_imgs = [nib.load(fname) for fname in args.in_labelmap_fnames]

    num_colors = len(label_imgs)
    cmap_name = args.cmap_name
    cmap = plt.get_cmap(cmap_name)

    # If the "Grays" cmap name is used, add one more color and drop the lightest
    # (white)
    _num_colors = num_colors
    _idx = 0
    if cmap_name == "Greys":
        _num_colors = num_colors + 1
        _idx = 1

    # Define the positions to sample colors from the colormap
    # Positions range from 0 to 1, where 0 is the start (lightest) and 1 is the end (darkest)
    positions = np.linspace(0, 1, _num_colors)[_idx:]
    # Sample colors from the colormap
    colors = [cmap(pos)[:3] for pos in positions]  # Drop the alpha channel

    opacity = 0.7
    linewidth = 4.5
    scene = plot_contour(
        ref_anat_img,
        label_imgs,
        args.label,
        colors,
        opacity=opacity,
        linewidth=linewidth,
    )

    size = (900, 900)
    window.snapshot(scene, fname=str(args.out_fname), size=size)


if __name__ == "__main__":
    main()
