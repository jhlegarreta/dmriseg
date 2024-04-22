#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute statistical significance between a reference contrast method and each
given contrast using the Wilcoxon rank-sum statistic.
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import ranksums

from dmriseg.data.lut.utils import class_id_label as lut_class_id_label
from dmriseg.utils.stat_preparation_utils import (
    pvalue_label,
    significance_label,
    statistic_label,
)


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "in_labels_fname",
        help="Labels filename (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "metric_name",
        help="Metric name",
        type=str,
    )
    parser.add_argument(
        "in_performance_fname_ref",
        help="Performance filename (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "--in_performance_fnames",
        nargs="+",
        help="Performance filenames (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "--out_fnames",
        nargs="+",
        help="Significance filenames (*.tsv)",
        type=Path,
    )
    return parser


def _parse_args(parser):

    args = parser.parse_args()

    return args


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    assert len(args.in_performance_fnames) == len(args.out_fnames)

    sep = "\t"

    df_lut = pd.read_csv(args.in_labels_fname, sep=sep)
    labels = df_lut[lut_class_id_label].values[1:]

    index_col = 0
    df_metric_ref = pd.read_csv(
        args.in_performance_fname_ref, sep=sep, index_col=index_col
    )
    metric_values_ref = df_metric_ref.values

    index = [statistic_label, pvalue_label]

    # Loop over statistics
    for in_performance_fname, out_fname in zip(
        args.in_performance_fnames, args.out_fnames
    ):

        df_metric = pd.read_csv(
            in_performance_fname, sep=sep, index_col=index_col
        )
        metric_values = df_metric.values

        res = ranksums(metric_values_ref, metric_values)

        df = pd.DataFrame(
            data=[res.statistic, res.pvalue], columns=labels, index=index
        )
        df.to_csv(
            out_fname, sep=sep, index=True, index_label=significance_label
        )


if __name__ == "__main__":
    main()
