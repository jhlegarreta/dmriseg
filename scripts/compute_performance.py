#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute performance.

Considerations:
- The participant ids have to match the prediction and ground truth data
filename sorting.
- Prediction filenames and ground truth filenames have to contain the participant
id.
"""

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
from dmriseg.io.utils import (
    build_suffix,
    participant_label_id,
    stats_fname_label,
    underscore,
)


def compute_measures(
    gnd_th_img, pred_img, spacing, labels, exclude_background
):

    _metrics = compute_metrics(
        gnd_th_img,
        pred_img,
        spacing,
        labels,
        exclude_background=exclude_background,
    )[0]
    vol_err = compute_volume_error(gnd_th_img, pred_img, labels)
    cm_dist, _, _ = compute_center_of_mass_distance(
        gnd_th_img, pred_img, labels
    )

    _metrics["vol_err"] = list(vol_err)
    _metrics["cm_dist"] = list(cm_dist)

    return _metrics


def create_measure_df(data, labels, sub_ids, describe=True):

    df = pd.DataFrame(data, columns=labels, index=sub_ids)
    df.index.name = participant_label_id
    # df.index.names = [idx_label]
    stats_df = None
    # Compute stats if requested
    if describe:
        with pd.option_context("mode.use_inf_as_na", True):
            stats_df = df.describe()

    return df, stats_df


def serialize_measures(metrics, measures, labels, sub_ids, sep, out_dirname):

    ext = DelimitedValuesFileExtension.TSV
    describe = True
    for measure in measures:
        _metric = np.asarray([item[measure.value] for item in metrics])
        df, stats_df = create_measure_df(
            _metric, labels, sub_ids, describe=describe
        )
        file_basename = measure.value
        fname = Path(out_dirname).joinpath(file_basename + build_suffix(ext))
        df.to_csv(fname, sep=sep, na_rep="NA")

        # Save stats
        if stats_df is not None:
            _basename = measure.value + underscore + stats_fname_label
            fname = Path(out_dirname).joinpath(_basename + build_suffix(ext))
            stats_df.to_csv(fname, sep=sep, na_rep="NA")


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

    assert len(gnd_th_lmap_fnames) == len(pred_lmap_fnames) == len(sub_ids)

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

        print(f"participant: {sub_id}")

        gnd_th_img = nib.load(gnd_th_fname)
        pred_img = nib.load(pred_fname)

        # Assert they have the same spacing
        gnd_th_img_spacing = gnd_th_img.header.get_zooms()
        pred_img_spacing = pred_img.header.get_zooms()

        assert np.allclose(gnd_th_img_spacing, pred_img_spacing)

        print(f"gnd_th_fname: {gnd_th_fname}")
        print(f"pred_fname: {pred_fname}")

        _metrics = compute_measures(
            gnd_th_img,
            pred_img,
            gnd_th_img_spacing,
            labels,
            exclude_background,
        )

        print("Computed metrics for participant")

        metrics.append(_metrics)

    # Get only the metrics of interest
    filtered_metrics = [
        {measure.value: item[measure.value] for measure in measures}
        for item in metrics
    ]

    # Serialize each measure to a different file
    print("Serializing")
    serialize_measures(
        filtered_metrics, measures, labels, sub_ids, sep, args.out_dirname
    )


if __name__ == "__main__":
    main()
