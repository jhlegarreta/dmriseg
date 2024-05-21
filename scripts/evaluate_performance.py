#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from dmriseg.analysis.measures import (
    Measure,
    compute_center_of_mass_distance,
    compute_metrics,
    compute_volume_error,
)
from dmriseg.data.lut.utils import class_id_label as lut_class_id_label
from dmriseg.io.file_extensions import DelimitedValuesFileExtension
from dmriseg.io.utils import build_suffix, participant_label_id, underscore

cm_dist_fname_label = "cm_dist_fname_label"
dice_fname_label = "dice"
hausdorff_fname_label = "hausdorff"
hausdorff95_fname_label = "hausdorff95"
jaccard_fname_label = "jaccard_fname_label"
msd_fname_label = "msd_fname_label"
vs_fname_label = "vs_fname_label"
stats_fname_label = "stats"


def create_measure_df(data, labels, sub_ids, describe=True):

    df = pd.DataFrame(
        data, columns=[participant_label_id, *labels], index=sub_ids
    )
    df.index = df[participant_label_id]
    # df.index.names = [idx_label]
    stats_df = None
    # Compute stats if requested
    if describe:
        stats_df = df.describe()

    return df, stats_df


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
        "in_gnd_th_labelmap_dirname",
        help="Ground truth labelmap dirname (*.nii.gz)",
        type=Path,
    )
    parser.add_argument(
        "in_pred_labelmap_dirname",
        help="Predicted labelmap dirname (*.nii.gz)",
        type=Path,
    )
    parser.add_argument(
        "in_labels_fname",
        help="Labels filename (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "out_dirname",
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

    gnd_th_lmap_fnames = [
        str(f)
        for f in sorted(
            Path(args.in_gnd_th_labelmap_dirname).rglob("*.nii.gz")
        )
    ]
    pred_lmap_fnames = [
        str(f)
        for f in sorted(Path(args.in_pred_labelmap_dirname).rglob("*.nii.gz"))
    ]

    sep = "\t"
    df_particip = pd.read_csv(args.in_participants_fname, sep=sep)
    sub_ids = sorted(df_particip[participant_label_id].values)

    df_lut = pd.read_csv(args.in_labels_fname, sep=sep)
    labels = sorted(df_lut[lut_class_id_label].values)

    exclude_background = True
    if exclude_background:
        labels = list(np.asarray(labels)[np.asarray(labels) != 0])

    measures = [
        Measure.DICE,
        Measure.JACCARD,
        Measure.HAUSDORFF,
        Measure.HAUSDORFF95,
        Measure.MEAN_SURFACE_DISTANCE,
        Measure.VOLUME_SIMILARITY,
        Measure.VOLUME_ERROR,
        Measure.CENTER_OF_MASS_DISTANCE,
    ]

    metrics = []
    for sub_id, gnd_th_fname, pred_fname in zip(
        sub_ids, gnd_th_lmap_fnames, pred_lmap_fnames
    ):

        assert str(sub_id) in Path(gnd_th_fname).with_suffix("").stem
        assert str(sub_id) in Path(pred_fname).with_suffix("").stem

        gnd_th_img = nib.load(gnd_th_fname)
        pred_img = nib.load(pred_fname)

        _metrics = compute_metrics(
            gnd_th_img, pred_img, labels, exclude_background=exclude_background
        )[0]
        vol_err = compute_volume_error(gnd_th_img, pred_img, labels)
        cm_dist, _, _ = compute_center_of_mass_distance(
            gnd_th_img, pred_img, labels
        )

        _metrics["vol_err"] = list(vol_err)
        _metrics["cm_dist"] = list(cm_dist)

        metrics.append(_metrics)

    # Get only the metrics of interest
    filtered_metrics = [
        {measure.value: item[measure.value] for measure in measures}
        for item in metrics
    ]

    # Serialize each measure to a different file
    ext = DelimitedValuesFileExtension.TSV
    describe = True
    for measure in measures:
        _metric = np.asarray(
            [item[measure.value] for item in filtered_metrics]
        )
        df, stats_df = create_measure_df(
            _metric, labels, sub_ids, describe=describe
        )
        file_basename = measure.value
        fname = Path(args.out_dirname).joinpath(
            file_basename + build_suffix(ext)
        )
        df.to_csv(fname, sep=sep, na_rep="NA")

        # Save stats
        if stats_df is not None:
            _basename = measure.value + underscore + stats_fname_label
            fname = Path(args.out_dirname).joinpath(
                _basename + build_suffix(ext)
            )
            stats_df.to_csv(fname, sep=sep)


if __name__ == "__main__":
    main()
