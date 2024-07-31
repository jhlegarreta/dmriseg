#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute labelmap volumes across participants and compute the label-wise
statistics.
"""

import argparse
import fnmatch
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from dmriseg.analysis.measures import compute_labelmap_volume
from dmriseg.data.lut.utils import class_id_label as lut_class_id_label
from dmriseg.io.utils import (
    append_label_to_fname,
    participant_label_id,
    stats_fname_label,
)

labelmap_dir_label = "suit_cer_seg_resized"


def find_files(directory, pattern):
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return sorted(matches)


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "in_gnd_th_labelmap_dirname",
        help="Ground truth labelmap dirname (*.nii.gz)",
        type=Path,
    )
    parser.add_argument(
        "in_participants_fname",
        help="Labels filename (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "in_labels_fname",
        help="Labels filename (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "out_filename",
        help="Output filename (*.tsv)",
        type=Path,
    )
    return parser


def _parse_args(parser):

    args = parser.parse_args()

    return args


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    sep = "\t"
    # pattern = "*__cer_seg_resized.nii.gz"
    # files = find_files(args.in_gnd_th_labelmap_dirname, pattern)

    df_lut = pd.read_csv(args.in_labels_fname, sep=sep)
    # Remove the background
    labels = df_lut[lut_class_id_label].values[1:10]  # [1:]

    participant_ids = pd.read_csv(args.in_participants_fname, sep=sep)[
        participant_label_id
    ].values

    df = pd.DataFrame()

    # Loop over participants
    for _id in participant_ids[:15]:

        # Retrieve the labelmap file
        path = args.in_gnd_th_labelmap_dirname / str(_id) / labelmap_dir_label
        fnames = [str(file) for file in path.iterdir() if file.is_file()]
        assert len(fnames) == 1

        image = nib.load(fnames[0])
        # Compute the volume for each label
        resolution = image.header.get_zooms()
        vol = compute_labelmap_volume(image.get_fdata(), labels, resolution)

        # Create a df with the volume data
        _df = pd.DataFrame(
            vol[np.newaxis, :], index=[_id], columns=list(map(str, labels))
        )
        _df.index.name = participant_label_id
        df = pd.concat([df, _df])

    # Save df
    df.to_csv(args.out_filename, sep=sep)
    # Compute stats across participants
    df_stats = df.describe()
    fname = append_label_to_fname(args.out_filename, stats_fname_label)
    df_stats.to_csv(fname, sep=sep)


if __name__ == "__main__":
    main()
