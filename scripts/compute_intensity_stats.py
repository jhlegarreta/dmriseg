#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import enum
import operator
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

participant_id_column = "participant_id"
filename_column = "filename"
totals_column = "total"
acceptance_column = "acceptance"


class IntensityRangeAcceptanceType(enum.Enum):
    ALL = "all"
    FOREGROUND_ONLY = "foreground_only"  # Considers foreground any value >= 0
    POSITIVE_ONLY = "positive_only"
    ROBUST = "robust"


def get_operator_for_acceptance(acceptance_type):

    if acceptance_type == IntensityRangeAcceptanceType.ALL:
        return None
    elif acceptance_type == IntensityRangeAcceptanceType.FOREGROUND_ONLY:
        return operator.gt
    elif acceptance_type == IntensityRangeAcceptanceType.POSITIVE_ONLY:
        return operator.ge
    else:
        raise ValueError("Not supported.")


def compute_acceptance_min_max(
    img_data, acceptance_type, min_percentile=2, max_percentile=98
):

    if acceptance_type == IntensityRangeAcceptanceType.ALL:
        lower_thr = np.min(img_data)
        upper_thr = np.max(img_data)
    elif acceptance_type == IntensityRangeAcceptanceType.FOREGROUND_ONLY:
        lower_thr = 0
        upper_thr = np.max(img_data)
    elif acceptance_type == IntensityRangeAcceptanceType.POSITIVE_ONLY:
        lower_thr = sys.float_info.epsilon
        upper_thr = np.max(img_data)
    elif acceptance_type == IntensityRangeAcceptanceType.ROBUST:
        lower_thr = np.percentile(img_data, min_percentile)
        # If the image contains too many background pixels the max percentile
        # may return zero, so use only the values above 0
        # if account_for_background:
        #    _intensities = img_data[img_data > 0]
        # else:
        #    _intensities = img_data
        _intensities = img_data[img_data > 0]
        upper_thr = np.percentile(_intensities, max_percentile)
    else:
        raise ValueError("Not supported.")

    return lower_thr, upper_thr


def plot_intensity_stats(intensities, figsize=(12, 10)):

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)

    _ = ax.boxplot(
        intensities,
        notch=True,
        patch_artist=True,
        meanline=True,
        showmeans=True,
    )
    fig.tight_layout()

    return fig


def compute_img_intensity(img, range_acceptance_type):

    # _operator = get_operator_for_acceptance(range_acceptance_type)
    img_data = img.get_fdata().flatten()
    # threshold = 0
    # if operator:
    #    return img_data[_operator(img_data, threshold)]
    # else:
    #     return img_data

    lower_thr, upper_thr = compute_acceptance_min_max(
        img_data, range_acceptance_type
    )

    # Trim values outside range
    return np.clip(img_data, lower_thr, upper_thr)


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("in_fname", help="Input filename (*.tsv).", type=Path)
    parser.add_argument(
        "out_data_fname",
        help="Output stats filename (*.tsv).",
        type=Path,
    )
    parser.add_argument(
        "out_plot_fname",
        help="Output plot filename (*.png).",
        type=Path,
    )
    parser.add_argument(
        "range_acceptance_type",
        help=f"Range_acceptance_type. Available: {list(IntensityRangeAcceptanceType.__members__)}",
        type=IntensityRangeAcceptanceType,
    )
    return parser


def _parse_args(parser):

    args = parser.parse_args()

    return args


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    df_fnames = pd.read_csv(args.in_fname, sep="\t")

    intensities = []

    range_acceptance_type = IntensityRangeAcceptanceType(
        args.range_acceptance_type
    )

    index = True
    _stats = []

    for _, row in df_fnames.iterrows():
        sub_id = row.loc[participant_id_column]
        fname = row.loc[filename_column]

        img = nib.load(fname)
        ntotals = np.size(img.get_fdata())
        _intensities = compute_img_intensity(img, range_acceptance_type)
        intensities.append(_intensities)

        df = pd.DataFrame(_intensities)
        _df_stats = df.describe().T
        _df_stats.insert(0, participant_id_column, sub_id)
        _df_stats.insert(1, totals_column, ntotals)
        _df_stats.insert(2, acceptance_column, range_acceptance_type.value)
        _df_stats.set_index(participant_id_column, inplace=True)
        _stats.append(_df_stats)

    df_stats = pd.concat(_stats)

    df_stats.to_csv(args.out_data_fname, index=index)

    # Randomly drop some lines since otherwise the data seems to require too
    # much memory to be plot/saved
    seed = 1234
    rng = np.random.default_rng(seed=seed)
    frac = 0.2
    sample_count = np.round(frac * len(df_fnames)).astype(int)
    choice_idx = rng.choice(len(intensities), sample_count, replace=False)
    _intensities = [intensities[i] for i in choice_idx]
    fig = plot_intensity_stats(_intensities)
    fig.show()
    fig.savefig(args.out_plot_fname)


if __name__ == "__main__":
    main()
