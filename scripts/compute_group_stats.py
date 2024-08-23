#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute performance statistics across groups of labels (e.g. all DCN, all
vermis, all lobules, all LH, RH).
Performance measure files are expected to have the following format:
``dmriseg.analysis.measuresMeasure.{TYPE}.value.tsv``
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dmriseg.analysis.measures import Measure
from dmriseg.data.lut.utils import get_diedrichsen_group_labels
from dmriseg.io.file_extensions import DelimitedValuesFileExtension
from dmriseg.io.utils import (
    build_suffix,
    group_fname_label,
    participant_label_id,
    stats_fname_label,
    underscore,
)


def compute_group_performance_statistics(
    dirname, measure, ext, sep, group_name
):

    suffix = build_suffix(ext)
    fname = dirname / Path(measure.value + suffix)
    df = pd.read_csv(fname, sep=sep, index_col=participant_label_id)

    # Select the labels corresponding to the group of interest and compute stats
    # across all elements
    labels = list(map(str, get_diedrichsen_group_labels(group_name)))
    df_subset = df[labels]

    # Compute stats
    # Replace inf for NaNs so that they do not skew the stats
    _df_subset = df_subset.replace([np.inf, -np.inf], np.nan, inplace=False)

    # Check if there is some row where all values are NaN (e.g. missed both
    # fastigial nuclei)
    idx = _df_subset[_df_subset.isnull().all(axis=1)].index
    if not idx.empty:
        print(
            f"{idx} contains all NaN values for {measure}; the segmentation probably failed for {group_name} labels"
        )

    # Compute the row-wise (e.g. participant-wise; across labels) stats, which
    # will yield a df with no NaN or np.inf
    _df_subset_row_wise_mean = pd.DataFrame(
        _df_subset.mean(axis=1), columns=[group_name]
    )

    # Check if all values (except for idx) are finite
    finite_check = np.isfinite(
        _df_subset_row_wise_mean.drop(idx, inplace=False)[
            group_name
        ].to_numpy()
    )

    # Assert that all values are finite
    assert (
        finite_check.all()
    ), f"{group_name}, {measure} contains non-finite values"

    # Compute stats across participants
    df_stats = _df_subset_row_wise_mean.describe()

    return df_stats


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "in_performance_dirname",
        help="Input dirname where performance data files dwell (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "out_dirname",
        help="Output dirname",
        type=Path,
    )
    parser.add_argument(
        "--group_names",
        help="Group names over which performances will be computed. If not "
        "specified, stats across all structures will be computed",
        nargs="+",
        type=str,
    )
    return parser


def _parse_args(parser):

    args = parser.parse_args()

    return args


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    measures = [
        Measure.CENTER_OF_MASS_DISTANCE,
        Measure.DICE,
        Measure.HAUSDORFF,
        Measure.HAUSDORFF95,
        Measure.JACCARD,
        Measure.MEAN_SURFACE_DISTANCE,
        Measure.VOLUME_ERROR,
        Measure.VOLUME_SIMILARITY,
        Measure.GT_LABEL_PRESENCE,
        Measure.PRED_LABEL_PRESENCE,
        Measure.LABEL_DETECTION_RATE,
    ]

    print(args.group_names)
    ext = DelimitedValuesFileExtension.TSV
    sep = "\t"

    group_names = args.group_names
    if group_names is None:
        group_names = ["all"]

    # Loop over measures
    for measure in measures:
        df_group_stats = pd.DataFrame()
        # Loop over group names
        for group_name in group_names:
            # Compute stats across all labels in group name
            _df_group_stats = compute_group_performance_statistics(
                args.in_performance_dirname, measure, ext, sep, group_name
            )

            # Concatenate to existing df horizontally
            df_group_stats = pd.concat(
                [df_group_stats, _df_group_stats], axis=1
            )

        _basename = (
            group_fname_label
            + underscore
            + measure.value
            + underscore
            + stats_fname_label
        )
        fname = Path(args.out_dirname).joinpath(_basename + build_suffix(ext))
        df_group_stats.to_csv(fname, sep=sep)


if __name__ == "__main__":
    main()
