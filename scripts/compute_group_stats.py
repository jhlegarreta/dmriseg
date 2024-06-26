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
    df_group = pd.DataFrame(
        df_subset.to_numpy().flatten(), columns=[group_name]
    )

    return df_group.describe()


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
