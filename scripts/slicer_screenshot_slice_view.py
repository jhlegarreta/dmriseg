"""
Capture a screenshot of a slice view in 3D Slicer.
"""

import argparse
import sys
import time
from pathlib import Path

import ScreenCapture
import slicer
import vtk
from PIL import Image


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


def capture_screenshot(view_name, offset, filename, width, height, dpi):

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

    # Reformat to requested size, height and DPI using PIL
    with Image.open(filename) as _img:
        img = resize_with_pad(_img, width, height)
        img.save(filename, dpi=(dpi, dpi))


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "in_filename",
        help="Input filename (*.nii.gz)",
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
    return parser


def _parse_args(parser):

    args = parser.parse_args()

    return args


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    # Load the input volume
    slicer.util.loadVolume(args.in_filename)

    width = 1920
    height = 1080
    dpi = 300

    set_layout(args.view_name)

    # For some reason, unless we introduce a delay, the image written is black.
    # The other solution is to call the script twice in a row from 3D Slicer.
    # Wait for some time
    delay = 0.500  # in seconds
    time.sleep(delay)

    capture_screenshot(
        args.view_name, args.offset, str(args.out_filename), width, height, dpi
    )

    # Exit so that 3D Slicer gets closed down
    sys.exit()


if __name__ == "__main__":
    main()
