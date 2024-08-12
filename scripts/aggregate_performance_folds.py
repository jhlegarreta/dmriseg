#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aggregate performance data across folds. Computes statistics after data has been
aggregated. No participant id duplicates should exist across folds and data
across folds is assumed to contain the same number of labels (columns).
Performance measure files are expected to have the following format:
``dmriseg.analysis.measuresMeasure.{TYPE}.value.tsv``
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dmriseg.analysis.measures import Measure
from dmriseg.io.file_extensions import DelimitedValuesFileExtension
from dmriseg.io.utils import (
    build_suffix,
    fold_label,
    participant_label_id,
    stats_fname_label,
    underscore,
)


def aggregate_data(dirnames, folds, measure, ext, sep):

    suffix = build_suffix(ext)
    fnames = [dirname / Path(measure.value + suffix) for dirname in dirnames]
    dfs = [
        pd.read_csv(fname, sep=sep, index_col=participant_label_id)
        for fname in fnames
    ]
    # Assert that they all have the same number of columns
    assert all(df.columns.equals(dfs[0].columns) for df in dfs)
    # Assert that they all have different participant ids
    participant_ids = sorted(np.hstack([df.index.values for df in dfs]))
    assert len(participant_ids) == len(set(participant_ids))

    # Add the fold name as a column for informative purposes
    [
        df.insert(loc=0, column=fold_label, value=fold)
        for fold, df in zip(folds, dfs)
    ]

    df_aggregate = pd.concat(dfs, ignore_index=False)
    df_aggregate.sort_values(participant_label_id, inplace=True)

    # Compute stats
    with pd.option_context("mode.use_inf_as_na", True):
        df_stats = df_aggregate.describe()

    return df_aggregate, df_stats


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--in_fold_performance_dirnames",
        help="Input dirnames where performance data files dwell (*.tsv)",
        nargs="+",
        type=Path,
    )
    parser.add_argument(
        "--fold_names",
        help="Fold name",
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "--out_dirname",
        help="Output dirname",
        type=Path,
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

    ext = DelimitedValuesFileExtension.TSV
    sep = "\t"

    # Compose and save the aggregate performance data file for each measure
    for measure in measures:
        df_aggregate, df_stats = aggregate_data(
            args.in_fold_performance_dirnames,
            args.fold_names,
            measure,
            ext,
            sep,
        )

        file_basename = measure.value
        fname = Path(args.out_dirname).joinpath(
            file_basename + build_suffix(ext)
        )
        df_aggregate.to_csv(fname, sep=sep, na_rep="NA")

        _basename = measure.value + underscore + stats_fname_label
        fname = Path(args.out_dirname).joinpath(_basename + build_suffix(ext))
        df_stats.to_csv(fname, sep=sep)


if __name__ == "__main__":
    main()
