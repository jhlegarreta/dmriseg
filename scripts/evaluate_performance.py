#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import enum
from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np
import pandas as pd

from dmriseg.analysis.measures import (
    compute_center_of_mass_distance,
    compute_surface_distance,
)

label_id = "ID"
particip_id = "ID"

fname_sep = "."
underscore = "_"
cm_dist_fname_label = "cm_dist_fname_label"
dice_fname_label = "dice"
hausdorff_fname_label = "hausdorff"
hausdorff95_fname_label = "hausdorff95"
jaccard_fname_label = "jaccard_fname_label"
msd_fname_label = "msd_fname_label"
vs_fname_label = "vs_fname_label"
stats_fname_label = "stats"


class Measures(enum.Enum):
    CENTER_OF_MASS_DISTANCE = "cm_dist"
    DICE = "dice"
    JACCARD = "jaccard"
    HAUSDORFF = "hausdorff"
    HAUSDORFF95 = "hausdorff95"
    MEAN_SURFACE_DISTANCE = "msd"
    VOLUME_SIMILARITY = "vs"


class DelimitedValuesFileExtension(enum.Enum):
    CSV = "csv"
    TSV = "tsv"


class CompressedFileExtension(enum.Enum):
    GZ = "gz"


def build_suffix(
    extension, compression: Union[None, CompressedFileExtension] = None
):

    if compression is None:
        return fname_sep + extension.value
    elif compression == CompressedFileExtension.GZ:
        return (
            fname_sep
            + extension.value
            + fname_sep
            + CompressedFileExtension.GZ.value
        )
    else:
        raise ValueError(f"Unknown compression: {compression}.")


def prepare_measure_filename(measure, dirname, ext):

    if measure == Measures.DICE:
        file_basename = dice_fname_label
    elif measure == Measures.JACCARD:
        file_basename = jaccard_fname_label
    elif measure == Measures.HAUSDORFF:
        file_basename = hausdorff_fname_label
    elif measure == Measures.HAUSDORFF95:
        file_basename = hausdorff95_fname_label
    elif measure == Measures.MEAN_SURFACE_DISTANCE:
        file_basename = msd_fname_label
    elif measure == Measures.VOLUME_SIMILARITY:
        file_basename = vs_fname_label
    elif measure == Measures.CENTER_OF_MASS_DISTANCE:
        file_basename = cm_dist_fname_label
    else:
        raise NotImplementedError(f"{measure} feature not expected.")

    return Path(dirname).joinpath(file_basename + build_suffix(ext))


def prepare_labeled_stats_filename(label, dirname, ext):

    _basename = label + underscore + stats_fname_label
    return Path(dirname).joinpath(_basename + build_suffix(ext))


def create_measure_df(data, labels, sub_ids, describe=True):

    df = pd.DataFrame(data, columns=labels, index=sub_ids)
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
    sub_ids = sorted(df_particip[particip_id].values)

    df_lut = pd.read_csv(args.in_labels_fname, sep=sep)
    labels = sorted(df_lut[label_id].values)

    exclude_background = True
    if exclude_background:
        labels = list(np.asarray(labels)[np.asarray(labels) != 0])

    measures = [
        Measures.DICE,
        Measures.JACCARD,
        Measures.HAUSDORFF,
        Measures.HAUSDORFF95,
        Measures.MEAN_SURFACE_DISTANCE,
        Measures.VOLUME_SIMILARITY,
        Measures.CENTER_OF_MASS_DISTANCE,
    ]

    metrics_dice = []
    metrics_jaccard = []
    metrics_hd = []
    metrics_hd95 = []
    metrics_msd = []
    metrics_vs = []
    cm_dist = []

    for _id, gnd_th_fname, pred_fname in zip(
        sub_ids, gnd_th_lmap_fnames, pred_lmap_fnames
    ):

        # assert _id in Path(gnd_th_fname).with_suffix("").stem
        # assert _id in Path(pred_fname).with_suffix("").stem

        gnd_th_img = nib.load(gnd_th_fname)
        pred_img = nib.load(pred_fname)

        _metrics = compute_surface_distance(
            gnd_th_img, pred_img, labels, exclude_background=exclude_background
        )[0]

        metrics_dice.append(_metrics["dice"])
        metrics_jaccard.append(_metrics["jaccard"])
        metrics_hd.append(_metrics["hd"])
        metrics_hd95.append(_metrics["hd95"])
        metrics_msd.append(_metrics["msd"])
        metrics_vs.append(_metrics["vs"])

        _cm_dist, _, _ = compute_center_of_mass_distance(
            gnd_th_img, pred_img, labels
        )
        cm_dist.append(_cm_dist)

    # ToDo
    # Rearrange to have all left/right next to each other

    ext = DelimitedValuesFileExtension.TSV
    describe = True
    measure_data = [
        metrics_dice,
        metrics_jaccard,
        metrics_hd,
        metrics_hd95,
        metrics_msd,
        metrics_vs,
        cm_dist,
    ]
    for measure, data in zip(measures, measure_data):

        df, stats_df = create_measure_df(
            np.asarray(data), labels, sub_ids, describe=describe
        )
        fname = prepare_measure_filename(measure, args.out_dirname, ext)
        df.to_csv(fname, sep=sep, na_rep="NA")

        # Save stats
        if stats_df is not None:
            fname = prepare_labeled_stats_filename(
                measure.value, args.out_dirname, ext
            )
            stats_df.to_csv(fname, sep=sep)

    # ToDo
    # Generate and save error measure plots


if __name__ == "__main__":
    main()
