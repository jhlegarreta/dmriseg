#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute a repeated measures ANOVA analysis.
"""

import argparse
import enum
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM

from dmriseg.analysis.measures import Measure
from dmriseg.data.lut.utils import (
    SuitAtlasDiedrichsenGroups,
    get_diedrichsen_group_labels,
)
from dmriseg.io.file_extensions import DelimitedValuesFileExtension
from dmriseg.io.utils import build_suffix, participant_label_id

contrast_label = "contrast"
subject_label = "subject"


class ContrastNames(enum.Enum):
    T1 = "t1"
    B0 = "b0"
    DWI = "dwi"
    DWI1k = "dwi1k"
    DWI2k = "dwi2k"
    DWI3k = "dwi3k"
    FA = "fa"
    MD = "md"
    RD = "rd"
    EVALS_E1 = "evalse1"
    EVALS_E2 = "evalse2"
    EVALS_E3 = "evalse3"
    AK = "ak"
    MK = "mk"
    RK = "rk"


def get_contrast_names_lut():
    return dict(
        {
            ContrastNames.T1.value: 1,
            ContrastNames.B0.value: 2,
            ContrastNames.DWI.value: 3,
            ContrastNames.DWI1k.value: 4,
            ContrastNames.DWI2k.value: 5,
            ContrastNames.DWI3k.value: 6,
            ContrastNames.FA.value: 7,
            ContrastNames.MD.value: 8,
            ContrastNames.RD.value: 9,
            ContrastNames.EVALS_E1.value: 10,
            ContrastNames.EVALS_E2.value: 11,
            ContrastNames.EVALS_E3.value: 12,
            ContrastNames.AK.value: 13,
            ContrastNames.MK.value: 14,
            ContrastNames.RK.value: 15,
        }
    )


def prepare_data_for_anova(dfs, measure, contrast_names):

    contrast_names_lut = get_contrast_names_lut()

    suit_labels = get_diedrichsen_group_labels(
        SuitAtlasDiedrichsenGroups.ALL.value
    )

    # Compute the mean across all labels for each participant/contrast
    columns_of_interest = list(map(str, suit_labels))

    measure_prtcpnt_mean = np.hstack(
        [df[columns_of_interest].mean(axis=1).values for df in dfs]
    )

    # Create the values for the participant (subject for AnovaRM) and contrast
    # (within for AnovaRM) columns
    participant_ids = np.hstack([df.index.to_numpy() for df in dfs])
    contrast = np.hstack(
        [
            len(df.index) * [contrast_names_lut[contrast_name]]
            for df, contrast_name in zip(dfs, contrast_names)
        ]
    )

    depvar_label = measure
    _subject_label = subject_label
    within_label = [contrast_label]

    df_anova = pd.DataFrame(
        {
            _subject_label: participant_ids,
            contrast_label: contrast,
            depvar_label: measure_prtcpnt_mean,
        }
    )

    return df_anova, depvar_label, _subject_label, within_label


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
