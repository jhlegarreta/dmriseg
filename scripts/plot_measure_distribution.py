#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot the distribution of measures for each contrast. Can be used to check
whether the repeated measures ANOVA assumption that the dependent variable (i.e.
the measure) is normally or approximately normally distributed.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dmriseg.analysis.measures import Measure
from dmriseg.io.file_extensions import DelimitedValuesFileExtension
from dmriseg.io.utils import build_suffix, participant_label_id
from dmriseg.utils.contrast_utils import get_contrast_names_lut
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
        "--labels",
        help="Subset of labels for the ANOVA analysis",
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "--out_fname",
        help="Output filename (*.png)",
        type=Path,
    )
    parser.add_argument(
        "--kde",
        help="Use KDE histogram representation",
        action="store_true",
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

    (
        df_anova,
        depvar_label,
        _subject_label,
        within_label,
    ) = prepare_data_for_anova(
        dfs,
        measure,
        args.contrast_names,
        columns_of_interest=list(map(str, args.labels)),
    )

    # Plot the data distribution
    # width_pixels = 1920
    # height_pixels = 1080
    # dpi = 300
    # figsize = (width_pixels / dpi, height_pixels / dpi)
    # plt.figure(figsize=figsize, dpi=dpi)

    sns.set(style="ticks")

    g = sns.displot(
        data=df_anova,
        x=f"{measure}",
        hue="contrast",
        kind="kde" if args.kde else "hist",
        palette="icefire",
    )

    plt.xlabel(measure)
    plt.ylabel("Distribution")

    # Rename the contrasts with their keys
    contrast_names_lut = get_contrast_names_lut()
    new_labels = contrast_names_lut.keys()
    for t, l in zip(g._legend.texts, new_labels):
        t.set_text(l)

    # Retrieve the current figure created by displot
    # fig = plt.gcf()

    # Set the size of the figure
    # fig.set_size_inches(figsize)
    plt.savefig(args.out_fname)


if __name__ == "__main__":
    main()
