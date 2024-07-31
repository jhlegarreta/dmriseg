#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot a labelmap as a set of 3D volumes.
"""

import argparse
from pathlib import Path

import numpy as np
import vtk

from dmriseg.data.lut.utils import read_lut_from_tsv2
from dmriseg.visualization.vtk_utils import (
    capture_vtk_render_window,
    render_labelmap_to_vtk,
    save_vtk_image,
)


def normalize_colors(lut):

    # Need colors to be in [0,1] for VTK
    cmap = {key: tuple(np.array(values) / 255) for key, values in lut.items()}

    return cmap


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
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
    parser.add_argument(
        "anatomical_view",
        help="Anatomical type",
        type=str,
    )
    return parser


def _parse_args(parser):

    args = parser.parse_args()

    return args


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    # Read the labelmap
    nifti_reader = vtk.vtkNIFTIImageReader()
    nifti_reader.SetFileName(args.in_labelmap_fname)
    nifti_reader.Update()

    labelmap = nifti_reader.GetOutput()

    lut = read_lut_from_tsv2(args.in_labels_fname)
    # Remove the background
    del lut[0]
    # Normalize colors
    normalized_colors = normalize_colors(lut)

    size = (1920, 1080)
    ren_win = render_labelmap_to_vtk(
        labelmap, normalized_colors, args.anatomical_view, size
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
