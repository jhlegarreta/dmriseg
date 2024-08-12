"""
Capture a screenshot of a slice view in 3D Slicer given a slice offset, zoom
factor and translation value. A segmentation mask can be optionally loaded: if
given, transparency is applied to the background pixels.
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


def get_slice_view_node(view_name):

    view_node_id = "vtkMRMLSliceNode" + view_name
    return slicer.mrmlScene.GetNodeByID(view_node_id)


def get_slice_view_node_label(view_name):

    if view_name == "axial":
        return "Red"
    elif view_name == "coronal":
        return "Green"
    elif view_name == "sagittal":
        return "Yellow"
    else:
        raise NotImplementedError(f"View {view_name} not implemented")


def set_layout(view_name):

    if view_name == "axial":
        layout_name = slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView
    elif view_name == "coronal":
        layout_name = slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpGreenSliceView
    elif view_name == "sagittal":
        layout_name = slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpYellowSliceView
    else:
        raise NotImplementedError(f"View {view_name} not implemented")

    layout_manager = slicer.app.layoutManager()
    layout_manager.setLayout(layout_name)


def disable_annotations():

    slice_annotations = (
        slicer.modules.DataProbeInstance.infoWidget.sliceAnnotations
    )
    slice_annotations.sliceViewAnnotationsEnabled = False
    slice_annotations.updateSliceViewFromGUI()


def get_transformation_element_idx(view_name):

    if view_name == "coronal":
        return 1, 3
    else:
        raise NotImplementedError(
            f"Transformation index for view {view_name} not implemented."
        )


def pan_slice(view_name, translation_row, translation_col, translation_value):
    slice_node = get_slice_view_node(view_name)

    # Get the slice-to-RASTransform matrix
    slice2ras_transform = slice_node.GetSliceToRAS()

    # Create a translation matrix (e.g., translate up by 10 mm in the Y direction)
    translationMatrix = vtk.vtkMatrix4x4()
    translationMatrix.SetElement(
        translation_row, translation_col, translation_value
    )

    # Apply the translation to the slice-to-RAS matrix
    vtk.vtkMatrix4x4.Multiply4x4(
        slice2ras_transform, translationMatrix, slice2ras_transform
    )

    # Update the slice-to-RAS transform
    slice_node.GetSliceToRAS().DeepCopy(slice2ras_transform)
    slice_node.UpdateMatrices()


def zoom_slice(view_name, zoom_factor):
    slice_node = get_slice_view_node(view_name)

    # Get the current field of view (FOV)
    current_fov = slice_node.GetFieldOfView()

    # Calculate the new FOV for zooming in
    new_fov = [fov / zoom_factor for fov in current_fov]

    # Apply the new FOV to the slice node
    slice_node.SetFieldOfView(new_fov[0], new_fov[1], new_fov[2])


def set_slice_offset(view_name, offset):
    slice_node = get_slice_view_node(view_name)
    slice_node.SetSliceOffset(offset)


def capture_screenshot(view_name, filename):

    cap = ScreenCapture.ScreenCaptureLogic()

    slice_node = get_slice_view_node(view_name)

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


def resize_image(fname, width, height, **params_pil):
    # Reformat to requested size, height and DPI using PIL
    with Image.open(fname) as _img:
        img = resize_with_pad(_img, width, height)
        img.save(fname, **params_pil)


def create_binary_image_from_file(fname):

    img = Image.open(fname).convert("L")

    # Threshold
    img = img.point(lambda p: 255 if p > 1 else 0)
    # Convert to monochrome
    return img.convert("1")


def close_binary_image(img):

    structure = np.ones((5, 5))
    closed_image = binary_closing(img, structure=structure)

    # Convert the result back to a PIL image
    return Image.fromarray((closed_image * 255).astype(np.uint8))


def create_mask_image_from_file(mask_scrnsht_fname):

    # Binarize the mask image
    binarized_img = create_binary_image_from_file(mask_scrnsht_fname)

    # Apply morphological closing to remove the border effects
    return close_binary_image(binarized_img)


def apply_mask_transparency(volume_scrnsht_fname, mask_scrnsht_fname):

    volume_img = Image.open(volume_scrnsht_fname).convert("RGBA")
    mask_img = Image.open(mask_scrnsht_fname).convert("L")  # grayscale
    volume_img.putalpha(mask_img)
    return volume_img


def mask_image(volume_scrnsht_fname, mask_scrnsht_fname):

    closed_image = create_mask_image_from_file(mask_scrnsht_fname)

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
        "zoom_factor",
        help="Zoom factor",
        type=float,
    )
    parser.add_argument(
        "translation_value",
        help="Translation value",
        type=float,
    )
    parser.add_argument(
        "--in_mask_filename",
        help="Input mask filename (*.nii.gz)",
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

    slice_view_node_name = get_slice_view_node_label(args.view_name)

    # Set the offset
    set_slice_offset(slice_view_node_name, args.offset)

    # Zoom and pan on the slice
    zoom_slice(slice_view_node_name, args.zoom_factor)
    col, row = get_transformation_element_idx(args.view_name)
    pan_slice(slice_view_node_name, col, row, args.translation_value)

    params_pil = {"dpi": (dpi, dpi)}
    capture_screenshot(slice_view_node_name, str(args.out_filename))
    resize_image(args.out_filename, width, height, **params_pil)

    # If a mask filename is given, set the background to transparent
    # ToDo
    # For some reason, if I follow the logic in the
    # `screenshot_slice_view_slicer.py` script, and clear the scene, load the
    # segmentation, set the layout, and zoom and pan, the mask screenshot does
    # not match the expected slice. If that were fixed, given the overlap
    # between these two scripts, a single script could be used for both
    # purposes.
    if args.in_mask_filename:

        # Load the mask
        slicer.util.loadSegmentation(args.in_mask_filename)
        set_layout(args.view_name)

        delay = 0.500  # in seconds
        time.sleep(delay)

        # Save the mask for visual checking purposes
        mask_fname = append_label_to_fname(args.out_filename, mask_label)
        capture_screenshot(slice_view_node_name, str(mask_fname))
        resize_image(mask_fname, width, height, **params_pil)

        # Apply transparency to the pixels outside the mask in the volume
        # screenshot
        volume_img = mask_image(args.out_filename, mask_fname)
        volume_img.save(args.out_filename, dpi=(dpi, dpi))

    # Exit so that 3D Slicer gets closed down
    sys.exit()


if __name__ == "__main__":
    main()
