#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot a labelmap as a set of 3D volumes. Can display a subset of the labels
depending on the input arguments; displays all labels if none specified.
"""

import argparse
from pathlib import Path

import numpy as np
import vtk

from dmriseg.data.lut.utils import (
    SuitAtlasDiedrichsenGroups,
    get_diedrichsen_group_labels,
    read_lut_from_tsv2,
)
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
    labels_group = parser.add_mutually_exclusive_group(required=False)
    labels_group.add_argument(
        "--group_name",
        help="SUIT label groups to be considered",
        type=str,
    )
    labels_group.add_argument(
        "--labels",
        help="SUIT labels to be considered",
        type=int,
        nargs="+",
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

    # Get the labels to be displayed
    if args.group_name is None and args.labels is None:
        # Display all labels
        group_name = SuitAtlasDiedrichsenGroups.ALL.value
        _labels = get_diedrichsen_group_labels(group_name)
    elif args.group_name is not None:
        _labels = get_diedrichsen_group_labels(args.group_name)
    elif args.labels is not None:
        _labels = args.labels
    else:
        raise ValueError("At least one label is required.")

    # Removes the background as well
    _lut = {k: v for k, v in lut.items() if k in _labels}

    # Normalize colors
    normalized_colors = normalize_colors(_lut)

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
