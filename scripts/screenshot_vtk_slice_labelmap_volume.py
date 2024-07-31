#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot a labelmap as a set of 3D volumes and a slice of the corresponding
anatomical image.
"""

import argparse
from pathlib import Path

import numpy as np
import vtk

from dmriseg.data.lut.utils import read_lut_from_tsv2
from dmriseg.visualization.vtk_utils import (
    capture_vtk_render_window,
    render_anat_labelmap_to_vtk,
    save_vtk_image,
)


def normalize_colors(lut):

    # Need colors to be in [0,1] for VTK
    cmap = {key: tuple(np.array(values) / 255) for key, values in lut.items()}

    return cmap


def read_mask(fname):

    mask_reader = vtk.vtkNIFTIImageReader()
    mask_reader.SetFileName(fname)
    mask_reader.Update()

    # Binarize
    threshold = vtk.vtkImageThreshold()
    threshold.SetInputConnection(mask_reader.GetOutputPort())
    threshold.ThresholdBetween(1, 1)
    threshold.ReplaceInOn()
    threshold.SetInValue(0)
    threshold.ReplaceOutOn()
    threshold.SetOutValue(1)
    threshold.Update()

    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputConnection(threshold.GetOutputPort())
    dmc.GenerateValues(1, 1, 1)
    dmc.Update()

    return dmc


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
        "in_labelmap_fname",
        help="Input labelmap filename (*.nii.gz)",
        type=Path,
    )
    parser.add_argument(
        "in_labels_fname",
        help="Labels filename (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "out_fname",
        help="Output filename (*.png)",
        type=Path,
    )
    return parser


def _parse_args(parser):

    args = parser.parse_args()

    return args


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    # Read the anatomical image
    nifti_reader_img = vtk.vtkNIFTIImageReader()
    nifti_reader_img.SetFileName(args.in_ref_anat_img_fname)
    nifti_reader_img.Update()
    anat_img = nifti_reader_img.GetOutput()

    # Read the labelmap
    nifti_reader_lmap = vtk.vtkNIFTIImageReader()
    nifti_reader_lmap.SetFileName(args.in_labelmap_fname)
    nifti_reader_lmap.Update()
    labelmap = nifti_reader_lmap.GetOutput()

    lut = read_lut_from_tsv2(args.in_labels_fname)
    # Remove the background
    del lut[0]
    # Normalize colors
    normalized_colors = normalize_colors(lut)

    size = (1920, 1080)
    ren_win = render_anat_labelmap_to_vtk(
        anat_img, labelmap, normalized_colors, size
    )

    # Do not display the rendering window
    ren_win.SetOffScreenRendering(True)
    # Trigger the rendering pipeline to set up the scene
    ren_win.Render()

    # Capture the VTK render window as an image
    image_data = capture_vtk_render_window(ren_win)

    # Save the image as a raster image (PNG)
    save_vtk_image(image_data, args.out_fname)


if __name__ == "__main__":
    main()
