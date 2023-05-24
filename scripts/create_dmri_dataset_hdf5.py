#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create dMRI segmentation HDF5 dataset file."""

import argparse
from pathlib import Path

import numpy as np
import yaml

from dmriseg.config.parsing_utils import parse_dataset_creation_config_file
from dmriseg.io.file_extensions import Compression
from dmriseg.io.hdf5_utils import create_hdf5_dataset


def _parse_args(cfg_fname):

    with open(cfg_fname) as f:
        cfg = yaml.safe_load(f.read())

    return parse_dataset_creation_config_file(cfg)


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "cfg_fname", help="Path to the config filename (*.yaml).", type=Path
    )

    return parser


def main():

    parser = _build_arg_parser()
    args = parser.parse_args()

    (
        dataset_name,
        learn_hdf5_dataset_fname,
        in_dirname,
        out_dirname,
        subj_split_fname,
        scalar_map,
        precision,
    ) = _parse_args(args.cfg_fname)

    # Assume all NIfTI files are compressed
    compression = Compression.GZ
    # When using np.float16 the error may be non-negligible; tested with some
    # values and the significant value is at the 4th decimal place for an FA
    # map. For np.float32, the error is 0.
    # Note that when using reduced precision, torch will operate also at such
    # precision, and conversion will be necessary.
    # Also, HDFView 3.1.0 is unable to display np.float16 data, despite the data
    # being there:
    # https://github.com/h5py/h5py/issues/1551
    # https://forum.hdfgroup.org/t/hdfview-3-1-0-error-displaying-dataset-with-np-float16-data/7974
    scalar_map_dtype = None
    # np.uint16 is not supported by torch:
    # return collate([torch.as_tensor(b) for b in batch], collate_fn_map=collate_fn_map)
    # TypeError: can't convert np.ndarray of type numpy.uint16.
    # The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.
    label_map_dtype = np.uint8

    # Create dataset
    create_hdf5_dataset(
        in_dirname,
        out_dirname,
        learn_hdf5_dataset_fname,
        subj_split_fname,
        dataset_name,
        scalar_map,
        compression,
        scalar_map_dtype,
        label_map_dtype,
    )


if __name__ == "__main__":
    main()
