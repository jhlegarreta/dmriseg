# -*- coding: utf-8 -*-

import numpy as np
from fury import actor

from dmriseg.anatomy.utils import Axis


# ToDo
# Adapt to RAS/LPS
def get_volume_axis_index(axis: Axis):
    """Assumes volume indices axes are sorted as sagittal, coronal, axial."""

    if axis == Axis.AXIAL:
        idx = 2
    elif axis == Axis.CORONAL:
        idx = 1
    elif axis == Axis.SAGITTAL:
        idx = 0

    return idx


def compute_central_slice(img_shape, axis: Axis):
    """."""

    idx = get_volume_axis_index(axis)
    return img_shape[idx] // 2


def compute_central_slices(img_shape):
    """."""

    # Get the relevant slices from the image
    x_slice = compute_central_slice(img_shape, Axis.SAGITTAL)
    y_slice = compute_central_slice(img_shape, Axis.CORONAL)
    z_slice = compute_central_slice(img_shape, Axis.AXIAL)

    return x_slice, y_slice, z_slice


def get_actor_extent(axis: Axis, img_shape, slice_index):
    """."""

    if axis == Axis.AXIAL:
        x1 = 0
        x2 = img_shape[0]
        y1 = 0
        y2 = img_shape[1]
        z1 = slice_index
        z2 = slice_index
    elif axis == Axis.CORONAL:
        x1 = 0
        x2 = img_shape[0]
        y1 = slice_index
        y2 = slice_index
        z1 = 0
        z2 = img_shape[1]
    elif axis == Axis.SAGITTAL:
        x1 = slice_index
        x2 = slice_index
        y1 = 0
        y2 = img_shape[1]
        z1 = 0
        z2 = img_shape[2]

    return x1, x2, y1, y2, z1, z2


def set_actor_display_extent(slice_actor, axis: Axis, img_shape, slice_index):
    """."""

    idx = get_volume_axis_index(axis)

    if slice_index is None:
        slice_index = img_shape[idx] // 2

    x1, x2, y1, y2, z1, z2 = get_actor_extent(axis, img_shape, slice_index)

    slice_actor.display_extent(x1, x2, y1, y2, z1, z2)


def _get_affine_for_texture(axis: Axis, offset):

    if axis == Axis.AXIAL:
        v = np.array([0.0, 0.0, offset])
    elif axis == Axis.CORONAL:
        v = np.array([0.0, -offset, 0.0])
    elif axis == Axis.SAGITTAL:
        v = np.array([offset, 0.0, 0.0])

    affine = np.identity(4)
    affine[0:3, 3] = v
    return affine


def create_volume_slice_actor(
    img,
    axis: Axis,
    slice_index,
    mask=None,
    value_range=None,
    offset=0.0,
    **kwargs,
):

    affine = _get_affine_for_texture(axis, offset)

    if mask is not None:
        img[np.where(mask == 0)] = 0

    if value_range:
        img = np.clip((img - value_range[0]) / value_range[1] * 255, 0, 255)

    slice_actor = actor.slicer(img, affine=affine, **kwargs)

    set_actor_display_extent(slice_actor, axis, img.shape, slice_index)
    # slice_actor = create_actor_view(slice_actor, slices, axis)

    return slice_actor


def create_volume_contour_actor(img, **kwargs):

    contour_actor = actor.contour_from_roi(img.get_fdata(), **kwargs)
    return contour_actor


def get_vertices_from_surface(surface):
    """Get vertices data from surface data.

    Parameters
    ----------
    surface : vtk.vtkPolyData or trimeshpy.TriMesh_Vtk
        Surface mesh.

    Returns
    -------
    vertices : ndarray(N, 3)
        Vertices.
    """

    import trimeshpy
    import vtk

    if isinstance(surface, vtk.vtkPolyData):
        vertices = trimeshpy.vtk_util.get_polydata_vertices(surface)
    elif isinstance(surface, trimeshpy.TriMesh_Vtk):
        vertices = surface.get_vertices()
    else:
        raise ValueError("Unsupported mesh type.")

    return vertices


def create_volume_contour_actor2(img, **kwargs):

    import vtk

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

    import os

    path = "/mnt/data/tractodata_testing_data/datasets/suit/"
    cereb_seg_fname = os.path.join(path, "cer_seg_100307_label_subset.nii")

    dmc = read_mask(cereb_seg_fname)
    from vtk.util import numpy_support

    from dmriseg.visualization.vtk_utils import smooth_polydata

    smoothed_polydata = smooth_polydata(dmc.GetOutput(), 10)
    cells = smoothed_polydata.GetPolys()
    nCells = cells.GetNumberOfCells()
    array = cells.GetData()
    assert array.GetNumberOfValues() % nCells == 0
    nCols = array.GetNumberOfValues() // nCells
    numpy_cells = numpy_support.vtk_to_numpy(array)
    numpy_cells = numpy_cells.reshape((-1, nCols))
    points = numpy_support.vtk_to_numpy(
        smoothed_polydata.GetPoints().GetData()
    )
    contour_actor = actor.surface(points, numpy_cells, smooth="butterfly")

    vertices = get_vertices_from_surface(smoothed_polydata)
    contour_actor = actor.surface(array, vertices, smooth="butterfly")

    # contour_actor = actor.contour_from_roi(img.get_fdata(), **kwargs)
    return contour_actor


def create_clip_actor(
    polydata_algorithm, color=None, opacity=0.7, linewidth=4.5
):
    """."""

    import vtk

    colors = vtk.vtkNamedColors()

    # If no color is provided, generate a default one
    if color is None:
        vtk_color3d = colors.GetColor3d("red")
    elif isinstance(color, str):
        vtk_color3d = colors.GetColor3d(color)
    elif isinstance(color, tuple):
        vtk_color3d = color
    else:
        raise ValueError(
            f"Color must be a string or tuple.\n"
            f"Provided an instance of: {color.__class__.__name__}"
        )

    # Actors and mappers
    clip_mapper = vtk.vtkPolyDataMapper()
    clip_mapper.ScalarVisibilityOff()
    clip_mapper.SetInputConnection(polydata_algorithm.GetOutputPort())

    clip_actor = vtk.vtkActor()
    clip_actor.GetProperty().SetDiffuse(0.0)
    clip_actor.GetProperty().SetAmbient(1.0)
    clip_actor.GetProperty().SetColor(vtk_color3d)
    clip_actor.GetProperty().SetOpacity(opacity)
    clip_actor.GetProperty().SetLineWidth(linewidth)  # will have no effect if
    # the polydata is not an edge
    clip_actor.SetMapper(clip_mapper)

    return clip_actor
