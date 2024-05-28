#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create participant folds randomly given the list of participant ids and the
number of folds using a k-fold splitting strategy. Splits ids across
train/valid/test sets. The number of folds defines the number of
train(+valid)/test splits, and the train/valid split is created from the
train(+valid) split by taking one unit less than the number of folds, and taking
the first split of the result.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import model_selection

from dmriseg.io.learning_description import LearningData, LearningSplit
from dmriseg.io.utils import (
    build_suffix,
    get_delimited_value_extension,
    participant_label_id,
)

split_label = LearningData.SPLIT.value
train_label = LearningSplit.TRAIN.value
valid_label = LearningSplit.VALID.value
test_label = LearningSplit.TEST.value


def create_folds(idx, num_splits, shuffle=None, random_state=None):

    kf_train_test = model_selection.KFold(
        n_splits=num_splits, shuffle=shuffle, random_state=random_state
    )
    folds = []

    # Create train(+valid)/test splits
    test_idx = []
    for _fold_id, (train_val_index, test_index) in enumerate(
        kf_train_test.split(idx)
    ):
        # Split train into train/valid: use the first fold for validation
        kf_train_valid = model_selection.KFold(
            n_splits=num_splits - 1, shuffle=shuffle, random_state=random_state
        )
        train_index, valid_index = list(kf_train_valid.split(train_val_index))[
            0
        ]
        print(f"Fold {_fold_id}:")
        print(f"train+valid:{train_val_index}")
        print(f"train: {train_val_index[train_index]}")
        print(f"valid: {train_val_index[valid_index]}")
        print(f"test: {test_index}")
        folds.append(
            (
                train_val_index[train_index],
                train_val_index[valid_index],
                test_index,
            )
        )
        test_idx.extend(test_index)

    # Ensure that all participants are at most once in the test set
    assert sorted(set(test_idx)) == list(idx)

    return folds


def split_participants(df, num_splits, shuffle=None, random_state=None):

    # Create folds
    idx = range(len(df))
    folds = create_folds(
        idx, num_splits, shuffle=shuffle, random_state=random_state
    )

    # Split participants
    df_folds = []
    for fold in folds:
        participant_ids = np.hstack(
            [
                df.iloc[fold[0]].values.flatten(),
                df.iloc[fold[1]].values.flatten(),
                df.iloc[fold[2]].values.flatten(),
            ]
        )
        labels = np.hstack(
            [
                len(fold[0]) * [train_label],
                len(fold[1]) * [valid_label],
                len(fold[2]) * [test_label],
            ]
        )
        df_folds.append(
            pd.DataFrame(
                zip(participant_ids, labels),
                columns=[participant_label_id, split_label],
            )
        )

    return df_folds


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "in_participant_data",
        help="Input participant data (*tsv)",
        type=Path,
    )
    parser.add_argument(
        "out_dirname",
        help="Output dirname for folds (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "folds",
        help="Number of folds",
        type=int,
        default=5,
    )
    return parser


def _parse_args(parser):

    args = parser.parse_args()

    return args


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    # Split the data randomly
    random_state = 1234
    shuffle = True

    # Read participant data
    sep = "\t"
    df = pd.read_csv(args.in_participant_data, sep=sep)

    # Create folds
    df_folds = split_participants(
        df, num_splits=args.folds, shuffle=shuffle, random_state=random_state
    )

    file_rootname = Path(args.in_participant_data).with_suffix("").stem

    ext = get_delimited_value_extension(sep)
    suffix = build_suffix(ext, compression=None)

    # Save folds
    for idx, df in enumerate(df_folds):
        out_dir_fold = args.out_dirname / f"fold-{idx}"
        os.makedirs(out_dir_fold, exist_ok=False)
        df.to_csv(
            out_dir_fold / f"{file_rootname}_fold-{idx}${suffix}",
            index=False,
            sep=sep,
        )
        # Save each split separately to ease things
        for split in [train_label, valid_label, test_label]:
            df_split = df.loc[df[split_label] == split][participant_label_id]
            df_split.to_csv(
                out_dir_fold
                / f"{file_rootname}_fold-{idx}_split-{split}${suffix}",
                index=False,
                sep=sep,
            )


if __name__ == "__main__":
    main()
