# -*- coding: utf-8 -*-

import enum
import os
from pathlib import Path
from typing import Union

import nibabel as nib
import nrrd
import numpy as np
import pandas as pd

from dmriseg.io.file_extensions import (
    Compression,
    NiftiDataExtension,
    NrrdDataExtension,
    PyTorchCheckpointExtension,
    fname_sep,
)
from dmriseg.io.study_description import SubjectData

checkpoint_file_rootname = "model"


class DiffusionScalarMapFilenamePattern(enum.Enum):
    FA = "fa"
    MD = "md"
    TRACE = "trace"


def build_suffix(extension, compression: Union[None, Compression] = None):

    if compression is None:
        return fname_sep + extension.value
    elif compression == Compression.GZ:
        return fname_sep + extension.value + fname_sep + Compression.GZ.value
    else:
        raise ValueError(f"Unknown compression: {compression}.")


def build_checkpoint_fname(dirname):

    return os.path.join(
        dirname,
        checkpoint_file_rootname + build_suffix(PyTorchCheckpointExtension.PT),
    )


def check_subject_data_inplace(df, in_dirname, scalar_map, compression):

    sub_id = df[SubjectData.ID.value].astype(str)

    for _sub_id in sub_id:

        dirname = os.path.join(in_dirname, _sub_id)

        assert check_dir_existence(dirname)

        # Check dMRI scalar maps
        for _scalar_map in scalar_map:
            # fname = os.path.join(dirname, _scalar_map + build_suffix(NiftiDataExtension.NII, compression=compression))
            # ToDo
            # Rename or fix the pattern
            file_rootname = _sub_id + "-dwi_b1000_" + _scalar_map
            fname = os.path.join(
                dirname,
                file_rootname
                + build_suffix(NrrdDataExtension.NRRD, compression=None),
            )
            assert check_file_existence(fname)

        # Check segmentations (labemaps)
        file_rootname = "wmparc_brain_mask"
        fname = os.path.join(
            dirname,
            file_rootname
            + build_suffix(NiftiDataExtension.NII, compression=compression),
        )
        assert check_file_existence(fname)


def check_dir_existence(dirname):
    return os.path.exists(dirname) and os.path.isdir(dirname)


def check_file_existence(fname):
    return os.path.exists(fname) and os.path.isfile(fname)


def read_learn_subject_data(fname):
    return pd.read_csv(fname)


def filter_filenames(fnames, labels):
    """Filter filenames based on the presence of a label.

    Parameters
    ----------
    fnames : list
        Filenames to be filtered.
    labels : list
        Labels whose presence is required.

    Returns
    -------
    filtered_fnames : list
        Filtered filenames.
    """

    filtered_fnames = [
        elem
        for elem in filter(
            lambda fname: any(fname for label in labels if label in fname),
            fnames,
        )
    ]

    return filtered_fnames


def get_image_file_type(fname):

    ext = Path(fname).suffixes[0][1:]

    if ext == NiftiDataExtension.NII.value:
        return NiftiDataExtension
    elif ext == NrrdDataExtension.NRRD.value:
        return NrrdDataExtension
    else:
        raise NotImplementedError(f"{ext} image file I/O not implemented.")


# ToDo
# Borrowed from
# https://github.com/pnlbwh/conversion/blob/master/conversion/nifti_write.py
def _space2ras(space):

    if len(space) == 3:
        # short definition LPI
        positive = [space[0], space[1], space[2]]

    else:
        # long definition left-posterior-inferior
        positive = space.split("-")

    xfrm = []
    if positive[0][0].lower() == "l":  # left
        xfrm.append(-1)
    else:
        xfrm.append(1)

    if positive[1][0].lower() == "p":  # posterior
        xfrm.append(-1)
    else:
        xfrm.append(1)

    if positive[2][0].lower() == "i":  # inferior
        xfrm.append(-1)
    else:
        xfrm.append(1)

    # Return 4x4 diagonal matrix
    xfrm.append(1)
    return np.diag(xfrm)


# ToDo
# Borrowed from
# https://github.com/pnlbwh/conversion/blob/master/conversion/nifti_write.py
# and stripped the 4D volume part
def nrrd2nifti(img_data, header):

    SPACE_UNITS = 2
    TIME_UNITS = 0

    SPACE2RAS = _space2ras(header["space"])

    translation = header["space origin"]

    rotation = header["space directions"]
    xfrm_nhdr = np.matrix(
        np.vstack(
            (
                np.hstack((rotation.T, np.reshape(translation, (3, 1)))),
                [0, 0, 0, 1],
            )
        )
    )

    xfrm_nifti = SPACE2RAS @ xfrm_nhdr
    # RAS2IJK= xfrm_nifti.I

    # automatically sets dim, data_type, pixdim, affine
    img_nifti = nib.nifti1.Nifti1Image(img_data, affine=xfrm_nifti)
    hdr_nifti = img_nifti.header

    # Now set xyzt_units, sform_code= qform_code= 2 (aligned)
    # https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/xyzt_units.html
    # Simplification assuming "mm" and "sec"
    hdr_nifti.set_xyzt_units(xyz=SPACE_UNITS, t=TIME_UNITS)
    hdr_nifti["qform_code"] = 2
    hdr_nifti["sform_code"] = 2

    hdr_nifti["descrip"] = "NRRD-->NIFTI transform by Tashrif Billah"
    img_nifti = nib.nifti1.Nifti1Image(
        img_data, affine=xfrm_nifti, header=hdr_nifti
    )
    return img_nifti


def retrieve_filename(dirname, label):

    file_basenames = [
        f
        for f in os.listdir(dirname)
        if os.path.isfile(os.path.join(dirname, f))
    ]
    file_basenames = filter_filenames(file_basenames, [label])

    assert len(file_basenames) == 1

    file_basename = file_basenames[0]

    return os.path.join(dirname, file_basename)


def read_image_data(fname):

    ext = get_image_file_type(fname)

    if ext == NiftiDataExtension:
        img = nib.load(fname)
        return img.get_fdata()
    elif ext == NrrdDataExtension:
        data, header = nrrd.read(fname)
        return nrrd2nifti(data, header).get_fdata()
    else:
        raise NotImplementedError(f"{ext} image file I/O not implemented.")


# ToDo
# Adapted from scilpy
def read_image_data_as_label_map(fname, dtype=np.uint16):

    allowed_label_types = [np.signedinteger, np.unsignedinteger]

    if any(map(lambda x: np.issubdtype(dtype, x), allowed_label_types)):
        img = nib.load(fname)
        return np.asanyarray(img.dataobj).astype(dtype)
    else:
        raise ValueError(
            f"The requested datatype {dtype} is not compatible with a label "
            f"image type. Allowed types are: {allowed_label_types}."
        )
