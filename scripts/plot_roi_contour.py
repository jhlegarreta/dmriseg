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
    create_masked_image_from_array,
    extract_roi_from_label_image,
)
from dmriseg.io.utils import append_label_to_fname, legend_label
from dmriseg.visualization.actor_utils import create_clip_actor
from dmriseg.visualization.plot_utils import (
    create_img_from_mpl_legend,
    create_mpl_legend,
    paste_image_into_canvas,
    rescale_image_keep_aspect,
)
from dmriseg.visualization.vtk_utils import compute_mask_planar_intersection


def plot_contour(
    ref_anat_img,
    label_imgs,
    labels,
    colors,
    opacity=0.7,
    linewidth=4.5,
    background=window.colors.white,
):

    # Extract ROI using the label of interest
    roi_img = [extract_roi_from_label_image(img, labels) for img in label_imgs]

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
        "--labels",
        help="Labels of interest",
        nargs="+",
        type=int,
    )
    parser.add_argument(
        "--in_labelmap_fnames",
        help="Labelmap image filename (*.nii.gz)",
        nargs="+",
        type=Path,
    )
    parser.add_argument(
        "--in_labelmap_names",
        help="Labelmap names",
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "--cmap_name", help="Colormap name", type=str, default="Greys"
    )
    parser.add_argument(
        "--transparent_bckgnd",
        action="store_true",
        help="Transparent background",
    )
    parser.add_argument(
        "--adjust_to_bbox", action="store_true", help="Adjust to bounding box"
    )
    return parser


def _parse_args(parser):

    args = parser.parse_args()

    return args


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    # Ensure that we have as many labelmap filenams as labelmap names
    assert len(args.in_labelmap_fnames) == len(args.in_labelmap_names)

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
    background = window.colors.white
    scene = plot_contour(
        ref_anat_img,
        label_imgs,
        args.labels,
        colors,
        opacity=opacity,
        linewidth=linewidth,
        background=background,
    )

    size = (1920, 1080)
    dpi = 300
    _out_fname = str(args.out_fname) if not args.transparent_bckgnd else None
    scene_array = window.snapshot(scene, fname=_out_fname, size=size, dpi=dpi)

    if args.transparent_bckgnd:
        # The scene array is an RGB image [0,255] so set the background value
        # accordingly from the FURY value
        _background = int(background[0] * 255)
        contour_img = create_masked_image_from_array(scene_array, _background)

        if args.adjust_to_bbox:
            bbox = contour_img.getbbox()
            cropped_img = contour_img.crop(bbox)
            rescaled_contour_img = rescale_image_keep_aspect(cropped_img, size)
            contour_img = paste_image_into_canvas(rescaled_contour_img, size)

        contour_img.save(args.out_fname, dpi=(dpi, dpi))

    # Create a legend to record the colors and save it to a file
    legend = create_mpl_legend(colors, args.in_labelmap_names, size, dpi)
    _lgnd_img = create_img_from_mpl_legend(legend, dpi)
    rescaled_lgnd_img = rescale_image_keep_aspect(_lgnd_img, size)
    lgnd_image = paste_image_into_canvas(rescaled_lgnd_img, size)

    legend_fname = append_label_to_fname(args.out_fname, legend_label)
    lgnd_image.save(legend_fname)


if __name__ == "__main__":
    main()
