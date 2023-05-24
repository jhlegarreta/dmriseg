#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Train the dMRI segmentation model."""

import argparse
from pathlib import Path

import yaml

from dmriseg.config.parsing_utils import parse_learning_config_file
from dmriseg.learning import learning_keys
from dmriseg.learning.dmriseg_learner import DMRISegLearner
from dmriseg.utils.torch_utils import prepare_device


def _parse_args(cfg_fname):

    with open(cfg_fname) as f:
        cfg = yaml.safe_load(f.read())

    return parse_learning_config_file(cfg)


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "cfg_fname", help="Path to the config filename (*.yaml).", type=Path
    )

    return parser


def main():

    parser = _build_arg_parser()
    args = parser.parse_args()

    (
        model_name,
        _model_kwargs,
        epochs,
        loss_func,
        _train_kwargs,
        learn_hdf5_dataset_fname,
        out_dirname,
        checkpoint_fname,
        optimizer_method,
        _optimizer_kwargs,
        lr_scheduler_type,
        _lr_scheduler_kwargs,
    ) = _parse_args(args.cfg_fname)

    device, kwargs = prepare_device()

    train_kwargs = dict({learning_keys.TRAIN_KWARGS: _train_kwargs})
    model_kwargs = dict({learning_keys.MODEL_KWARGS: _model_kwargs})
    optimizer_kwargs = dict(
        {learning_keys.OPTIMIZER_KWARGS: _optimizer_kwargs}
    )
    lr_scheduler_kwargs = dict(
        {learning_keys.LR_SCHEDULER_KWARGS: _lr_scheduler_kwargs}
    )

    kwargs = dict(
        train_kwargs, **model_kwargs, **optimizer_kwargs, **lr_scheduler_kwargs
    )

    # Train
    learner = DMRISegLearner(
        learn_hdf5_dataset_fname,
        out_dirname,
        model_name,
        device,
        loss_func,
        epochs,
        optimizer_method,
        lr_scheduler_type,
        **kwargs,
    )
    learner.run()


if __name__ == "__main__":
    main()
