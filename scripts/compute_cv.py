#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute the coefficient of variation (CV) of the input data per each label in
the provided labelmap.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from dmriseg.data.lut.utils import (
    SuitAtlasDiedrichsenGroups,
    class_id_label,
    get_diedrichsen_group_labels,
)
from dmriseg.io.file_extensions import DelimitedValuesFileExtension
from dmriseg.io.utils import (
    build_suffix,
    fold_label,
    group_fname_label,
    participant_label_id,
    stats_fname_label,
    underscore,
)

label_label = "label"
mean_label = "mean"
std_label = "std"
cv_label = "cv"

folds = ("fold-0", "fold-1", "fold-2", "fold-3", "fold-4")

fold_fielname = "cerebellum_participants_hcp_qc_no_sub_prefix"


def compute_cv_per_label(img, labelmap, labels):

    img_data = img.get_fdata()
    label_data = labelmap.get_fdata()

    results = []

    for label in labels:
        mask = label_data == label
        values = img_data[mask]
        mean = np.mean(values) if any(values) else np.nan
        std = np.std(values) if any(values) else np.nan
        cv = std / mean if mean != 0 else np.nan

        results.append(
            {
                class_id_label: int(label),
                mean_label: mean,
                std_label: std,
                cv_label: cv,
            }
        )

    df = pd.DataFrame(results)
    df.set_index(class_id_label, inplace=True)

    return df


def _compose_participant_fold_df(in_participants_fold_dirname):

    df_sub_fold = pd.DataFrame()

    # Compose the df containing ids and folds
    for fold in folds:
        fname = f"{in_participants_fold_dirname}/{fold}/cerebellum_participants_hcp_qc_no_sub_prefix_{fold}_split-test.tsv"
        df = pd.read_csv(fname)
        # Add the fold information
        df[fold_label] = [fold] * len(df)

        df_sub_fold = pd.concat([df_sub_fold, df], axis=0)

    df_sub_fold.sort_values(by=[participant_label_id], inplace=True)

    return df_sub_fold


def _build_fnames(
    in_dwi_scalarmap_dirname,
    in_pred_labelmap_root_dirname,
    contrast_folder_label,
    pred_fname_label,
    dti_metric,
    fold,
    sub_id,
):

    scalarmap_fname = f"{in_dwi_scalarmap_dirname}/{sub_id}/resize_dtimetrics_dwi/{sub_id}__{dti_metric}_resized.nii.gz"
    labelmap_fname = f"{in_pred_labelmap_root_dirname}/{contrast_folder_label}/{fold}/results/prediction/{sub_id}__{pred_fname_label}_pred.nii.gz"

    return scalarmap_fname, labelmap_fname


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "in_participants_fname",
        help="Labels filename (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "in_participants_fold_dirname",
        help="Participants fold dirname",
        type=Path,
    )
    parser.add_argument(
        "in_dwi_scalarmap_dirname",
        help="Input DWI scalarmap (FA, MD, etc.) dirname (*.nii.gz)",
        type=Path,
    )
    parser.add_argument(
        "in_pred_labelmap_root_dirname",
        help="Predicted labelmap dirname (*.nii.gz)",
        type=Path,
    )
    parser.add_argument(
        "contrast_folder_label",
        help="Contrast folder label",
        type=str,
    )
    parser.add_argument(
        "pred_fname_label",
        help="Prediction file label",
        type=str,
    )
    parser.add_argument(
        "dti_metric",
        help="DTI metric (FA, MD, etc.)",
        type=str,
    )
    parser.add_argument(
        "in_labels_fname",
        help="Labels filename (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "out_dirname",
        help="Output dirname (*.tsv)",
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
    df_particip = pd.read_csv(args.in_participants_fname, sep=sep)
    sub_ids = sorted(df_particip[participant_label_id].values)

    df_lut = pd.read_csv(args.in_labels_fname, sep=sep)
    labels = sorted(df_lut[class_id_label].values)

    exclude_background = True
    if exclude_background:
        labels = list(np.asarray(labels)[np.asarray(labels) != 0])

    df_sub_fold = _compose_participant_fold_df(
        args.in_participants_fold_dirname
    )

    assert np.alltrue(df_sub_fold[participant_label_id].values == sub_ids)

    df_mean = pd.DataFrame()
    df_std = pd.DataFrame()
    df_cv = pd.DataFrame()

    # Loop over each participant
    for sub_id in sub_ids:

        fold = df_sub_fold[df_sub_fold[participant_label_id] == sub_id][
            "fold"
        ].values
        assert len(fold) == 1
        fold = fold[0]

        scalarmap_fname, labelmap_fname = _build_fnames(
            args.in_dwi_scalarmap_dirname,
            args.in_pred_labelmap_root_dirname,
            args.contrast_folder_label,
            args.pred_fname_label,
            args.dti_metric,
            fold,
            sub_id,
        )

        img = nib.load(scalarmap_fname)
        labelmap = nib.load(labelmap_fname)

        # Missing labels are dealt with implicitly: a missing label in the
        # labelmap (prediction) will not get any value; when the df gets
        # concatenated, the missing label will get a NaN value.
        df = compute_cv_per_label(img, labelmap, labels)

        _df_mean = pd.DataFrame(
            [df[mean_label].values],
            index=[
                sub_id,
            ],
            columns=df.index,
        )
        _df_std = pd.DataFrame(
            [df[std_label].values],
            index=[
                sub_id,
            ],
            columns=df.index,
        )
        _df_cv = pd.DataFrame(
            [df[cv_label].values],
            index=[
                sub_id,
            ],
            columns=df.index,
        )

        df_mean = pd.concat([df_mean, _df_mean])
        df_std = pd.concat([df_std, _df_std])
        df_cv = pd.concat([df_cv, _df_cv])

    sep = "\t"
    index = True
    na_rep = "NA"
    ext = DelimitedValuesFileExtension.TSV

    _basename = args.dti_metric + underscore + mean_label
    fname = Path(args.out_dirname).joinpath(_basename + build_suffix(ext))

    df_mean.to_csv(fname, index=index, sep=sep, na_rep=na_rep)

    _basename = args.dti_metric + underscore + std_label
    fname = Path(args.out_dirname).joinpath(_basename + build_suffix(ext))

    df_std.to_csv(fname, index=index, sep=sep, na_rep=na_rep)

    _basename = args.dti_metric + underscore + cv_label
    fname = Path(args.out_dirname).joinpath(_basename + build_suffix(ext))

    df_cv.to_csv(fname, index=index, sep=sep, na_rep=na_rep)

    # Compute label-wise stats (across participants)
    stats_mean_df = df_mean.describe()
    stats_std_df = df_std.describe()
    stats_cv_df = df_cv.describe()

    _basename = (
        args.dti_metric
        + underscore
        + mean_label
        + underscore
        + stats_fname_label
        + underscore
        + label_label
    )
    fname = Path(args.out_dirname).joinpath(_basename + build_suffix(ext))

    stats_mean_df.to_csv(fname, index=index, sep=sep, na_rep=na_rep)

    _basename = (
        args.dti_metric
        + underscore
        + std_label
        + underscore
        + stats_fname_label
        + underscore
        + label_label
    )
    fname = Path(args.out_dirname).joinpath(_basename + build_suffix(ext))

    stats_std_df.to_csv(fname, index=index, sep=sep, na_rep=na_rep)

    _basename = (
        args.dti_metric
        + underscore
        + cv_label
        + underscore
        + stats_fname_label
        + underscore
        + label_label
    )
    fname = Path(args.out_dirname).joinpath(_basename + build_suffix(ext))

    stats_cv_df.to_csv(fname, index=index, sep=sep, na_rep=na_rep)

    # Compute group stats (across label groups)
    group_names = [
        SuitAtlasDiedrichsenGroups.ALL,
        SuitAtlasDiedrichsenGroups.DCN,
        SuitAtlasDiedrichsenGroups.DENTATE,
        SuitAtlasDiedrichsenGroups.INTERPOSED,
        SuitAtlasDiedrichsenGroups.FASTIGIAL,
        SuitAtlasDiedrichsenGroups.VERMIS,
        SuitAtlasDiedrichsenGroups.LOBULES,
        SuitAtlasDiedrichsenGroups.CRUS,
    ]

    group_mean_df = pd.DataFrame()
    group_std_df = pd.DataFrame()
    group_cv_df = pd.DataFrame()

    for group_name in group_names:

        labels = get_diedrichsen_group_labels(group_name.value)
        group_mean_series = df_mean[labels].stack().describe()
        group_std_series = df_std[labels].stack().describe()
        group_cv_series = df_cv[labels].stack().describe()

        group_mean_series.rename(group_name.value, inplace=True)
        group_std_series.rename(group_name.value, inplace=True)
        group_cv_series.rename(group_name.value, inplace=True)

        group_mean_df = pd.concat([group_mean_df, group_mean_series], axis=1)
        group_std_df = pd.concat([group_std_df, group_std_series], axis=1)
        group_cv_df = pd.concat([group_cv_df, group_cv_series], axis=1)

    _basename = (
        args.dti_metric
        + underscore
        + mean_label
        + underscore
        + stats_fname_label
        + underscore
        + group_fname_label
    )
    fname = Path(args.out_dirname).joinpath(_basename + build_suffix(ext))

    group_mean_df.to_csv(fname, index=index, sep=sep, na_rep=na_rep)

    _basename = (
        args.dti_metric
        + underscore
        + std_label
        + underscore
        + stats_fname_label
        + underscore
        + group_fname_label
    )
    fname = Path(args.out_dirname).joinpath(_basename + build_suffix(ext))

    group_std_df.to_csv(fname, index=index, sep=sep, na_rep=na_rep)

    _basename = (
        args.dti_metric
        + underscore
        + cv_label
        + underscore
        + stats_fname_label
        + underscore
        + group_fname_label
    )
    fname = Path(args.out_dirname).joinpath(_basename + build_suffix(ext))

    group_cv_df.to_csv(fname, index=index, sep=sep, na_rep=na_rep)


if __name__ == "__main__":
    main()
