#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute statistical significance between the performance of contrast methods
using a pairwise (Student's) t-test.
"""

import argparse
from pathlib import Path

import pandas as pd
import pingouin as pg

from dmriseg.io.file_extensions import DelimitedValuesFileExtension
from dmriseg.io.utils import build_suffix, participant_label_id, underscore
from dmriseg.stats.utils import StatisticalTest
from dmriseg.utils.stat_preparation_utils import prepare_data_for_pairwise_test


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
        "measure_name",
        help="Measure name",
        type=str,
    )
    parser.add_argument(
        "out_dirname",
        help="Output dirname",
        type=Path,
    )
    parser.add_argument(
        "--in_performance_fnames",
        nargs="+",
        help="Performance filenames (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "--in_contrast_names",
        nargs="+",
        help="Contrast names",
        type=str,
    )
    return parser


def _parse_args(parser):

    args = parser.parse_args()

    return args


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    assert len(args.in_performance_fnames) == len(args.in_contrast_names)

    sep = "\t"
    ext = DelimitedValuesFileExtension.TSV

    stat_test = StatisticalTest.PAIRWISE_T_TEST.value

    fnames = [fname for fname in args.in_performance_fnames]

    dfs = [
        pd.read_csv(fname, sep=sep, index_col=participant_label_id)
        for fname in fnames
    ]

    # Assert we have the same participants for all dfs (contrasts)
    # Compute the difference with respect to the first
    assert not any(
        [
            set(dfs[0].index.values.tolist()).difference(
                set(df.index.values.tolist())
            )
            for df in dfs[1:]
        ]
    )

    (
        data_df,
        depvar_label,
        _subject_label,
        within_label,
    ) = prepare_data_for_pairwise_test(
        dfs, args.measure_name, args.in_contrast_names
    )

    # When parametric=True, the below call is equivalent to
    # pingouin.pairwise_ttests()
    # See Enhancements in https://pingouin-stats.org/build/html/changelog.html#v0-5-2-june-2022
    #
    parametric = True
    padjust = "fdr_bh"
    stats_df = pg.pairwise_tests(
        data=data_df,
        dv=depvar_label,
        within=within_label,
        subject=_subject_label,
        parametric=parametric,
        padjust=padjust,
    )

    _basename = args.measure_name + underscore + stat_test
    fname = Path(args.out_dirname).joinpath(_basename + build_suffix(ext))
    stats_df.to_csv(fname, sep=sep)


if __name__ == "__main__":
    main()
