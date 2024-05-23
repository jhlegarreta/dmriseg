# -*- coding: utf-8 -*-

import numpy as np
import scipy.ndimage as ndimage
import vtk
from fury.utils import set_input
from vtk.util import numpy_support

from dmriseg.visualization.utils import compute_plane_normal


def build_vtk_transform_from_np_affine(affine):

    transform = vtk.vtkTransform()
    transform_matrix = vtk.vtkMatrix4x4()
    transform_matrix.DeepCopy(
        (
            affine[0][0],
            affine[0][1],
            affine[0][2],
            affine[0][3],
            affine[1][0],
            affine[1][1],
            affine[1][2],
            affine[1][3],
            affine[2][0],
            affine[2][1],
            affine[2][2],
            affine[2][3],
            affine[3][0],
            affine[3][1],
            affine[3][2],
            affine[3][3],
        )
    )
    transform.SetMatrix(transform_matrix)
    transform.Inverse()

    return transform


def reslice_image(img, affine=None):

    nb_components = 1

    data = (img > 0) * 1
    vol = np.interp(data, xp=[data.min(), data.max()], fp=[0, 255])
    vol = vol.astype("uint8")

    im = vtk.vtkImageData()
    di, dj, dk = vol.shape[:3]
    im.SetDimensions(di, dj, dk)
    voxsz = (1.0, 1.0, 1.0)
    # im.SetOrigin(0, 0, 0)
    im.SetSpacing(voxsz[2], voxsz[0], voxsz[1])
    im.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, nb_components)

    # copy data
    vol = np.swapaxes(vol, 0, 2)
    vol = np.ascontiguousarray(vol)

    vol = vol.ravel()

    uchar_array = numpy_support.numpy_to_vtk(vol, deep=0)
    im.GetPointData().SetScalars(uchar_array)

    # Set the affine to the identity if none given
    if affine is None:
        affine = np.eye(4)

    # Create the transform
    transform = build_vtk_transform_from_np_affine(affine)

    # Set the reslicing
    resliced_img = vtk.vtkImageReslice()
    set_input(resliced_img, im)
    resliced_img.SetResliceTransform(transform)
    resliced_img.AutoCropOutputOn()

    # Adding this will allow to support anisotropic voxels
    # and also gives the opportunity to slice per voxel coordinates

    rzs = affine[:3, :3]
    zooms = np.sqrt(np.sum(rzs * rzs, axis=0))
    resliced_img.SetOutputSpacing(*zooms)

    resliced_img.SetInterpolationModeToLinear()
    resliced_img.Update()

    return resliced_img


def extract_contour(mask_img, affine=None, feature_angle=0.0):

    resliced_img = reslice_image(mask_img, affine=affine)

    contour_filter = vtk.vtkContourFilter()
    contour_filter.SetInputData(resliced_img.GetOutput())

    contour_filter.SetValue(0, 1)

    contour_normals = vtk.vtkPolyDataNormals()
    contour_normals.SetInputConnection(contour_filter.GetOutputPort())
    contour_normals.SetFeatureAngle(feature_angle)

    return contour_normals


def create_clipping_plane(img, anatomical_view):
    """."""

    # Make the clipping plane be at center of mass
    # ref_anat_img_center_of_mass = ndimage.measurements.center_of_mass(
    #    np.array(np.asanyarray(ref_anat_img.dataobj)))
    # tractogram_center_of_mass = ndimage.measurements.center_of_mass(
    #    tractogram_mask
    # )

    normal = compute_plane_normal(anatomical_view)
    origin = ndimage.measurements.center_of_mass(img.get_fdata())

    clip_plane = vtk.vtkPlane()
    clip_plane.SetNormal(normal)
    clip_plane.SetOrigin(origin)

    return clip_plane


def compute_mask_planar_intersection(
    mask, clip_plane, affine=None, fill_contour=False
):
    """."""

    # Extract the contour of the mask
    contour_normals = extract_contour(mask, affine=affine)

    # Clip the source with the plane
    # clipper = vtk.vtkClipPolyData()
    # clipper.SetInputConnection(contour_normals.GetOutputPort())
    # clipper.SetClipFunction(clip_plane)

    # Edges
    # feature_edges = vtk.vtkFeatureEdges()
    # feature_edges.SetInputConnection(clipper.GetOutputPort())
    # feature_edges.BoundaryEdgesOn()
    # feature_edges.FeatureEdgesOff()
    # feature_edges.ManifoldEdgesOff()
    # feature_edges.NonManifoldEdgesOff()
    # feature_edges.ColoringOff()
    # feature_edges.Update()

    cutter = vtk.vtkCutter()
    cutter.SetInputConnection(contour_normals.GetOutputPort())
    cutter.SetCutFunction(clip_plane)
    cutter.Update()

    polydata_algorithm = cutter

    if fill_contour:
        contour_triangulator = vtk.vtkContourTriangulator()
        contour_triangulator.SetInputConnection(cutter.GetOutputPort())
        contour_triangulator.Update()

        polydata_algorithm = contour_triangulator

    # return feature_edges
    return polydata_algorithm


def smooth_polydata(polydata, iterations=15, rlx_factor=0.1):
    """
    http://www.vtk.org/doc/nightly/html/classvtkSmoothPolyDataFilter.html
    """

    smooth_filter = vtk.vtkSmoothPolyDataFilter()
    smooth_filter.SetInputData(polydata)
    smooth_filter.SetNumberOfIterations(iterations)
    smooth_filter.SetRelaxationFactor(rlx_factor)
    smooth_filter.FeatureEdgeSmoothingOff()
    # smooth_filter.FeatureEdgeSmoothingOn()
    smooth_filter.BoundarySmoothingOff()
    # smooth_filter.BoundarySmoothingOff()
    smooth_filter.Update()
    return smooth_filter.GetOutput()


def smooth_polydata2(poly, iterations=15, feature_angle=120, pass_band=0.001):
    """
    http://www.vtk.org/doc/nightly/html/classvtkWindowedSincPolyDataFilter.html#details
    """
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(poly)
    smoother.SetNumberOfIterations(iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(feature_angle)
    smoother.SetPassBand(pass_band)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return smoother.GetOutput()


def extract_polydata_from_image_data(vtk_image_data):

    threshold = vtk.vtkImageThreshold()
    threshold.SetInputData(vtk_image_data)
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

    return dmc.GetOutput()
