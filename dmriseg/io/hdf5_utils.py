# -*- coding: utf-8 -*-

import os

import h5py

from dmriseg.config import hdf5_dataset_keys
from dmriseg.io.learning_description import LearningData, LearningSplit
from dmriseg.io.study_description import SubjectData
from dmriseg.io.utils import (
    DiffusionScalarMapFilenamePattern,
    check_subject_data_inplace,
    read_image_data,
    read_image_data_as_label_map,
    read_learn_subject_data,
    retrieve_filename,
)


def create_hdf5_dataset(
    in_dirname,
    out_dirname,
    hdf5_dataset_fname,
    subj_split_fname,
    dataset_name,
    scalar_map,
    compression,
    scalar_map_dtype,
    label_map_dtype,
):

    df = read_learn_subject_data(subj_split_fname)
    check_subject_data_inplace(df, in_dirname, scalar_map, compression)

    # Create the HDF5 file
    fname = os.path.join(out_dirname, hdf5_dataset_fname)
    with h5py.File(fname, "w") as f:

        dataset_group = f.create_group(dataset_name)

        # ToDo
        # Write attributes
        # write_anatomy_as_attributes(dataset_group, dataset_name)

        subj_group = dataset_group.create_group(
            hdf5_dataset_keys.SUBJECT_GROUP
        )

        group = subj_group.create_group(hdf5_dataset_keys.TRAIN_GROUP)
        sub_id = get_subject_list(df, LearningSplit.TRAIN.value)
        write_learning_split(
            group,
            sub_id,
            in_dirname,
            scalar_map,
            scalar_map_dtype,
            label_map_dtype,
        )

        group = subj_group.create_group(hdf5_dataset_keys.VALID_GROUP)
        sub_id = get_subject_list(df, LearningSplit.VALID.value)
        write_learning_split(
            group,
            sub_id,
            in_dirname,
            scalar_map,
            scalar_map_dtype,
            label_map_dtype,
        )

        group = subj_group.create_group(hdf5_dataset_keys.TEST_GROUP)
        sub_id = get_subject_list(df, LearningSplit.TEST.value)
        write_learning_split(
            group,
            sub_id,
            in_dirname,
            scalar_map,
            scalar_map_dtype,
            label_map_dtype,
        )


def get_subject_list(df, split):
    return (
        df.groupby([LearningData.SPLIT.value])
        .get_group(split)[SubjectData.ID.value]
        .values.astype(str)
        .tolist()
    )


def write_learning_split(
    group, sub_id, in_dirname, scalar_map, scalar_map_dtype, label_map_dtype
):

    # Loop over subjects
    for _sub_id in sub_id:

        subject_group = group.create_group(_sub_id)

        sub_dirname = os.path.join(in_dirname, _sub_id)
        write_subject_data(
            subject_group,
            sub_dirname,
            scalar_map,
            scalar_map_dtype,
            label_map_dtype,
        )


def write_subject_data(
    group, in_dirname, scalar_map, scalar_map_dtype, label_map_dtype
):

    diffusion_group = group.create_group(hdf5_dataset_keys.DIFFUSION_GROUP)
    write_diffusion_scalar_map_data(
        diffusion_group, in_dirname, scalar_map, scalar_map_dtype
    )

    structural_group = group.create_group(hdf5_dataset_keys.STRUCTURAL_GROUP)
    write_segmentation_data(structural_group, in_dirname, label_map_dtype)


def write_diffusion_scalar_map_data(group, in_dirname, scalar_map, dtype):

    scalar_map_group = group.create_group(hdf5_dataset_keys.SCALAR_MAP_GROUP)

    # Loop over the scalar maps
    for _scalar_map in scalar_map:

        fname = retrieve_filename(in_dirname, _scalar_map)
        data = read_image_data(fname).astype(dtype)
        name = get_scalar_map_hdf5_key(_scalar_map)
        scalar_map_group.create_dataset(name, data=data)


def get_scalar_map_hdf5_key(scalar_map):
    if scalar_map == DiffusionScalarMapFilenamePattern.FA.value:
        return hdf5_dataset_keys.FA_DATA
    elif scalar_map == DiffusionScalarMapFilenamePattern.MD.value:
        return hdf5_dataset_keys.MD_DATA
    elif scalar_map == DiffusionScalarMapFilenamePattern.TRACE.value:
        return hdf5_dataset_keys.TRACE_DATA
    else:
        raise ValueError(f"Unknown scalar map: {scalar_map}.")


def write_segmentation_data(group, in_dirname, label_map_dtype):

    label_map_group = group.create_group(hdf5_dataset_keys.LABEL_MAP_GROUP)

    label = "wmparc_brain_mask"
    fname = retrieve_filename(in_dirname, label)
    data = read_image_data_as_label_map(fname, dtype=label_map_dtype)

    name = hdf5_dataset_keys.SEGMENTATION_DATA
    label_map_group.create_dataset(name, data=data)
