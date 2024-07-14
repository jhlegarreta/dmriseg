"""
Capture a screenshot of a slice view in 3D Slicer. A segmentation can be
optionally loaded, and a color LUT be applied to the segmentation. If a
segmentation mask is given, transparency is applied to the background pixels.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import ScreenCapture
import slicer
import vtk
from PIL import Image
from scipy.ndimage import binary_closing

mask_label = "mask"
underscore = "_"


def append_label_to_fname(fname, label):

    dirname = fname.parent
    file_rootname = fname.stem
    suffix = fname.suffix
    file_basename = Path(file_rootname + underscore + label + suffix)
    return dirname / file_basename


def set_layout(view_name):

    if view_name == "Green":
        layout_name = slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpGreenSliceView
    else:
        raise NotImplementedError()

    layout_manager = slicer.app.layoutManager()
    layout_manager.setLayout(layout_name)


def get_slice_view_node(view_name):

    view_node_id = "vtkMRMLSliceNode" + view_name
    return slicer.mrmlScene.GetNodeByID(view_node_id)


def disable_annotations():

    slice_annotations = (
        slicer.modules.DataProbeInstance.infoWidget.sliceAnnotations
    )
    slice_annotations.sliceViewAnnotationsEnabled = False
    slice_annotations.updateSliceViewFromGUI()


def resize_with_pad(image, target_width, target_height, color="black"):
    """Resize PIL image keeping ratio and using a canvas with the specified
    background color.
    """

    target_ratio = target_height / target_width
    im_ratio = image.height / image.width
    if target_ratio > im_ratio:
        # It must be fixed by width
        resize_width = target_width
        resize_height = round(resize_width * im_ratio)
    else:
        # Fixed by height
        resize_height = target_height
        resize_width = round(resize_height / im_ratio)

    image_resize = image.resize(
        (resize_width, resize_height),
        Image.Resampling.LANCZOS,
    )
    background = Image.new(
        "RGBA",
        (target_width, target_height),
        color=color,
    )
    offset = (
        round((target_width - resize_width) / 2),
        round((target_height - resize_height) / 2),
    )
    background.paste(image_resize, offset)
    return background.convert("RGB")


def capture_screenshot(view_name, offset, filename):

    cap = ScreenCapture.ScreenCaptureLogic()

    slice_node = get_slice_view_node(view_name)
    slice_node.SetSliceOffset(offset)

    # Allow some time to change the slice
    delay = 0.750  # in seconds
    time.sleep(delay)

    view = cap.viewFromNode(slice_node)

    disable_annotations()

    # Capture the screenshot
    render_window = view.renderWindow()
    render_window.SetAlphaBitPlanes(1)
    wti = vtk.vtkWindowToImageFilter()
    wti.SetInputBufferTypeToRGBA()
    wti.SetInput(render_window)
    wti.Update()

    # Write the VTK image data
    vtk_data = wti.GetOutput()
    writer = vtk.vtkPNGWriter()
    writer.SetWriteToMemory(False)
    writer.SetInputData(vtk_data)
    writer.SetFileName(filename)
    writer.Write()


def resize_image(fname, width, height, **params_pil):
    # Reformat to requested size, height and DPI using PIL
    with Image.open(fname) as _img:
        img = resize_with_pad(_img, width, height)
        img.save(fname, **params_pil)


def binarize_image(fname):

    img = Image.open(fname).convert("L")

    # Threshold
    img = img.point(lambda p: 255 if p > 1 else 0)
    # Convert to monochrome
    return img.convert("1")


def perform_morphological_closing(img):

    structure = np.ones((5, 5))
    closed_image = binary_closing(img, structure=structure)

    # Convert the result back to a PIL image
    return Image.fromarray((closed_image * 255).astype(np.uint8))


def process_mask_image(mask_scrnsht_fname):

    # Binarize the mask image
    binarized_img = binarize_image(mask_scrnsht_fname)

    # Apply morphological closing to remove the border effects
    return perform_morphological_closing(binarized_img)


def apply_mask_transparency(volume_scrnsht_fname, mask_scrnsht_fname):

    volume_img = Image.open(volume_scrnsht_fname).convert("RGBA")
    mask_img = Image.open(mask_scrnsht_fname).convert("L")  # grayscale
    volume_img.putalpha(mask_img)
    return volume_img


def process_image(volume_scrnsht_fname, mask_scrnsht_fname):

    closed_image = process_mask_image(mask_scrnsht_fname)

    # Save the result
    closed_image.save(mask_scrnsht_fname)

    return apply_mask_transparency(volume_scrnsht_fname, mask_scrnsht_fname)


def load_data(volume_fname, segmentation_fname=None, lut_fname=None):

    # Load the input volume
    slicer.util.loadVolume(volume_fname)

    # Load the LUT before the segmentation
    color_table = None
    if lut_fname:
        color_table = slicer.util.loadColorTable(lut_fname)

    # Load the segmentation file
    if segmentation_fname:
        properties = {}
        if color_table:
            properties = {"colorNodeID": color_table.GetID()}
        slicer.util.loadSegmentation(segmentation_fname, properties=properties)


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "in_volume_filename",
        help="Input volume filename (*.nii.gz)",
        type=Path,
    )
    parser.add_argument(
        "out_filename",
        help="Output filename (*.png)",
        type=Path,
    )
    parser.add_argument(
        "view_name",
        help="View name",
        type=str,
    )
    parser.add_argument(
        "offset",
        help="Offset",
        type=float,
    )
    parser.add_argument(
        "--in_segmentation_filename",
        help="Input segmentation filename (*.nii.gz)",
        type=Path,
    )
    parser.add_argument(
        "--in_mask_filename",
        help="Input mask filename (*.nii.gz)",
        type=Path,
    )
    parser.add_argument(
        "--in_color_lut_filename",
        help="Input color LUT table filename (*.txt)",
        type=Path,
    )
    return parser


def _parse_args(parser):

    args = parser.parse_args()

    return args


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    load_data(
        args.in_volume_filename,
        segmentation_fname=args.in_segmentation_filename,
        lut_fname=args.in_color_lut_filename,
    )

    width = 1920
    height = 1080
    dpi = 300

    set_layout(args.view_name)

    # For some reason, unless we introduce a delay, the image written is black.
    # The other solution is to call the script twice in a row from 3D Slicer.
    # Wait for some time
    delay = 0.500  # in seconds
    time.sleep(delay)

    params_pil = {"dpi": (dpi, dpi)}
    capture_screenshot(args.view_name, args.offset, str(args.out_filename))
    resize_image(args.out_filename, width, height, **params_pil)

    # If a mask filename is given, set the background to transparent
    if args.in_mask_filename:
        # Clean the scene
        slicer.mrmlScene.Clear(0)

        # Load the mask
        slicer.util.loadSegmentation(args.in_mask_filename)
        # Removing the outline does not improve the outline effect that is seen
        # in the captured screenshot
        # segmentation_node = slicer.util.loadSegmentation(args.in_mask_filename)
        # segmentation_disp_node = segmentation_node.GetDisplayNode()
        # segmentation_disp_node.SetVisibility2DOutline(False)

        set_layout(args.view_name)

        delay = 0.500  # in seconds
        time.sleep(delay)

        # Save the mask for visual checking purposes
        mask_fname = append_label_to_fname(args.out_filename, mask_label)
        capture_screenshot(args.view_name, args.offset, str(mask_fname))
        resize_image(mask_fname, width, height, **params_pil)

        # Apply transparency to the pixels outside the mask in the volume
        # screenshot
        volume_img = process_image(args.out_filename, mask_fname)
        volume_img.save(args.out_filename, dpi=(dpi, dpi))

    # Exit so that 3D Slicer gets closed down
    sys.exit()


if __name__ == "__main__":
    main()
