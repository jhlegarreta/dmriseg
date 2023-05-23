# -*- coding: utf-8 -*-

from dmriseg.config import config_keys


def parse_dataset_creation_io_data(cfg):

    io_group = cfg[config_keys.IO]
    learn_hdf5_dataset_fname = io_group[config_keys.LEARN_HDF5_DATASET_FNAME]
    in_dirname = io_group[config_keys.IN_DIRNAME]
    out_dirname = io_group[config_keys.OUT_DIRNAME]
    subj_split_fname = io_group[config_keys.SUBJ_SPLIT_FNAME]

    return learn_hdf5_dataset_fname, in_dirname, out_dirname, subj_split_fname


def parse_dataset_creation_dataset_data(cfg):
    return cfg[config_keys.DATASET][config_keys.NAME]


def parse_dataset_creation_config_file(cfg):

    dataset_name = parse_dataset_creation_dataset_data(cfg)

    (
        learn_hdf5_dataset_fname,
        in_dirname,
        out_dirname,
        subj_split_fname,
    ) = parse_dataset_creation_io_data(cfg)

    scalar_map = cfg[config_keys.SCALAR_MAP]
    precision = cfg[config_keys.PRECISION]

    return (
        dataset_name,
        learn_hdf5_dataset_fname,
        in_dirname,
        out_dirname,
        subj_split_fname,
        scalar_map,
        precision,
    )


def parse_model_data(cfg):

    model_group = cfg[config_keys.MODEL]
    model_name = model_group.pop(config_keys.NAME)
    return model_name, model_group


def parse_training_data(cfg):

    train_group = cfg[config_keys.TRAIN]
    epochs = train_group.pop(config_keys.EPOCHS)
    loss_func = train_group.pop(config_keys.LOSS_FUNCTION)
    return epochs, loss_func, train_group


def parse_learning_io_data(cfg):

    io_group = cfg[config_keys.IO]
    learn_hdf5_dataset_fname = io_group[config_keys.LEARN_HDF5_DATASET_FNAME]
    out_dirname = io_group[config_keys.OUT_DIRNAME]
    checkpoint_fname = io_group[config_keys.CHECKPOINT_FNAME]
    return learn_hdf5_dataset_fname, out_dirname, checkpoint_fname


def parse_optimizer(cfg):
    optimizer_group = cfg[config_keys.OPTIMIZER]
    method = optimizer_group.pop(config_keys.METHOD)
    lr_scheduler_group = optimizer_group.pop(config_keys.LR_SCHEDULER)
    lr_scheduler_type = lr_scheduler_group.pop(config_keys.TYPE)
    return method, optimizer_group, lr_scheduler_type, lr_scheduler_group


def parse_learning_config_file(cfg):
    model_name, model_kwargs = parse_model_data(cfg)
    epochs, loss_func, train_kwargs = parse_training_data(cfg)
    (
        learn_hdf5_dataset_fname,
        out_dirname,
        checkpoint_fname,
    ) = parse_learning_io_data(cfg)
    (
        optimizer_method,
        optimizer_kwargs,
        lr_scheduler_type,
        lr_scheduler_kwargs,
    ) = parse_optimizer(cfg)

    return (
        model_name,
        model_kwargs,
        epochs,
        loss_func,
        train_kwargs,
        learn_hdf5_dataset_fname,
        out_dirname,
        checkpoint_fname,
        optimizer_method,
        optimizer_kwargs,
        lr_scheduler_type,
        lr_scheduler_kwargs,
    )
