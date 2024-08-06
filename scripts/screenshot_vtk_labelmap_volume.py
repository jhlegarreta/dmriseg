#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot a labelmap as a set of 3D volumes. Can display a subset of the labels
depending on the input arguments; displays all labels if none specified.

If an anatomical reference image is given, it will also plot a slice of the
image. Additionally, if a mask is given, it will mask the slice so that only the
area within the mask is shown, the rest being set to full transparency.
"""

import argparse
from pathlib import Path

import vtk

from dmriseg.data.lut.utils import (
    SuitAtlasDiedrichsenGroups,
    get_diedrichsen_group_labels,
    read_lut_from_tsv2,
)
from dmriseg.io.utils import (
    append_label_to_fname,
    mask_fname_label,
    masked_fname_label,
)
from dmriseg.visualization.color_utils import normalize_colors
from dmriseg.visualization.utils import mask_image
from dmriseg.visualization.vtk_utils import (
    capture_vtk_render_window,
    render_labelmap_to_vtk,
    render_volume_slice_to_vtk,
    save_vtk_image,
)


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
    parser.add_argument(
        "--in_ref_anat_img_fname",
        help="Reference anatomical image filename (*.nii.gz)",
        type=Path,
    )
    parser.add_argument(
        "--mask_fname",
        help="Mask image filename (*.nii.gz)",
        type=Path,
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
    nifti_reader_lmap = vtk.vtkNIFTIImageReader()
    nifti_reader_lmap.SetFileName(args.in_labelmap_fname)
    nifti_reader_lmap.Update()
    labelmap = nifti_reader_lmap.GetOutput()

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

    anat_img = None
    size = (1920, 1080)
    if args.in_ref_anat_img_fname:
        # Read the anatomical image
        nifti_reader_img = vtk.vtkNIFTIImageReader()
        nifti_reader_img.SetFileName(args.in_ref_anat_img_fname)
        nifti_reader_img.Update()
        anat_img = nifti_reader_img.GetOutput()

    ren_win = render_labelmap_to_vtk(
        labelmap,
        normalized_colors,
        args.anatomical_view,
        size,
        anat_img=anat_img,
    )

    # Do not display the rendering window
    ren_win.SetOffScreenRendering(True)
    # Trigger the rendering pipeline to set up the scene
    ren_win.Render()

    # Capture the VTK render window as an image
    image_data = capture_vtk_render_window(ren_win)

    # Save the image as a raster image (PNG)
    save_vtk_image(image_data, args.out_fname)

    # Mask the slice
    if args.mask_fname:
        dpi = 300

        # ToDo
        # The approach below:
        #
        # Create a volume -> Create a surface -> Define a cutting plane ->
        # Cut the surface to get the intersection -> Fill contours -> Render
        #
        # i.e.
        #
        # vtkImageThreshold (upper, lower thr to 1) ->
        # vtkDiscreteMarchingCubes -> vtkPlane (e.g. normal to 0,0,1;
        # origin to 0,0,140 for an axial slice) -> vtkCutter ->
        # vtkContourTriangulator -> vtkPolyDataMapper -> Actor, Renderer, Window
        #
        # to create a render window on the mask slice of interest to use it as
        # a mask to set to full transparency the volume/labelmap slice area
        # outside the mask does not play well with the cutting plane origin,
        # etc. and the parameters used for the slice in the
        # `vtk_utils.slice_vtk_image` method, and the camera settings in
        # `vtk_utls.render_labelmap_to_vtk`, so for now use the exact same
        # approach used to generate the volume slice but using the mask, and
        # use PIL to apply the transparency to the slice using the mask.
        #

        # Read the mask image
        nifti_reader_mask = vtk.vtkNIFTIImageReader()
        nifti_reader_mask.SetFileName(args.mask_fname)
        nifti_reader_mask.Update()
        mask_img = nifti_reader_mask.GetOutput()

        # Render the mask to VTK
        _ren_win = render_volume_slice_to_vtk(mask_img, size)

        # Do not display the rendering window
        _ren_win.SetOffScreenRendering(True)
        # Trigger the rendering pipeline to set up the scene
        _ren_win.Render()

        # Capture the VTK render window as an image
        _image_data = capture_vtk_render_window(_ren_win)

        # Save the image as a raster image (PNG)
        mask_fname = append_label_to_fname(args.out_fname, mask_fname_label)
        save_vtk_image(_image_data, mask_fname)

        # Read both files with PIL and apply transparency
        volume_img = mask_image(args.out_fname, mask_fname)
        masked_img = append_label_to_fname(args.out_fname, masked_fname_label)
        volume_img.save(masked_img, dpi=(dpi, dpi))


if __name__ == "__main__":
    main()
