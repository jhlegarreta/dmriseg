#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute a repeated measures ANOVA analysis.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM

from dmriseg.analysis.measures import Measure
from dmriseg.io.file_extensions import DelimitedValuesFileExtension
from dmriseg.io.utils import build_suffix, participant_label_id
from dmriseg.utils.stat_preparation_utils import prepare_data_for_anova


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--in_performance_dirnames",
        help="Input dirname where performance data files dwell (*.tsv)",
        type=Path,
        nargs="+",
    )
    parser.add_argument(
        "--contrast_names", help="Contrast names", type=str, nargs="+"
    )
    parser.add_argument(
        "--measure_name",
        help="Measure name",
        type=str,
    )
    parser.add_argument(
        "--out_fname",
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

    assert len(args.in_performance_dirnames) == len(args.contrast_names)

    ext = DelimitedValuesFileExtension.TSV
    sep = "\t"

    measure = Measure(args.measure_name).value

    suffix = build_suffix(ext)
    file_basename = measure + suffix

    # Get all relevant files
    fnames = [
        dirname / file_basename for dirname in args.in_performance_dirnames
    ]
    assert np.all([fname.is_file() for fname in fnames])

    dfs = [
        pd.read_csv(fname, sep=sep, index_col=participant_label_id)
        for fname in fnames
    ]

    # For Dice coefficients, we cannot have NaN of inf values
    assert not any([df.isna().any().any() for df in dfs])
    assert not any(
        [
            np.isinf(df.select_dtypes(include=[np.number]).values).any()
            for df in dfs
        ]
    )

    (
        df_anova,
        depvar_label,
        _subject_label,
        within_label,
    ) = prepare_data_for_anova(dfs, measure, args.contrast_names)

    # Conduct the repeated measures ANOVA
    aov = AnovaRM(
        data=df_anova,
        depvar=depvar_label,
        subject=_subject_label,
        within=within_label,
    ).fit()

    print(aov)

    aov.anova_table.to_csv(args.out_fname, sep=sep)


if __name__ == "__main__":
    main()
