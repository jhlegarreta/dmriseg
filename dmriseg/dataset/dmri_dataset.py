# -*- coding: utf-8 -*-

import enum

import h5py
from torch.utils import data

from dmriseg.config import hdf5_dataset_keys


class LearningDataParts(enum.Enum):
    IMAGE = "image"
    LABELMAP = "labelmap"


class DMRISegDataset(data.Dataset):
    def __init__(self, dataset_fname, learning_split):

        self._dataset_fname = dataset_fname
        self._learning_split = learning_split
        self._length = 0

        self._images = []
        self._labelmaps = []
        self._sub_ids = []

        self.load_data()

    # def __del__(self):
    #    self._hdf5_file.close()

    def __getitem__(self, idx):
        # ToDo
        # Selectively load the requested scalar maps as channels
        img = self._images[idx]
        labelmap = self._labelmaps[idx]
        # For now, expand dimensions to incorporate a single channel
        # PyTorch uses the channel first convention
        img = img[None, :, :]
        labelmap = labelmap[None, :, :]

        return {
            LearningDataParts.IMAGE.value: img,
            LearningDataParts.LABELMAP.value: labelmap,
        }

    def __len__(self):
        return self._length

    def load_data(self):

        with h5py.File(self._dataset_fname, "r") as f:

            # Assume a single dataset exists
            dataset_name = list(f.keys())[0]
            sub_id = self._get_subject_ids(f, dataset_name)

            for _sub_id in sub_id:

                self._sub_ids.extend(_sub_id)

                # ToDo
                # Selectively load the requested scalar maps
                fa_data = self._get_scalar_map_data(
                    f, dataset_name, _sub_id, hdf5_dataset_keys.FA_DATA
                )
                self._images.extend(fa_data)

                labelmap_data = self._get_label_map_data(
                    f, dataset_name, _sub_id
                )
                self._labelmaps.extend(labelmap_data)

        self._length = len(self._images)

    def _get_data(
        self,
        hdf5_file,
        dataset_name,
        sub_id,
        mri_data_group,
        map_type,
        data_type,
    ):
        dataset_group = hdf5_file[dataset_name]
        sub_group = dataset_group[hdf5_dataset_keys.SUBJECT_GROUP]
        learning_split = sub_group[self._learning_split.value]
        mri_data_group = learning_split[sub_id][mri_data_group]
        map_group = mri_data_group[map_type]
        # [()] gets the whole as a 3D array; list() formats it as a list of 2D
        # arrays
        # return map_group[data_type][()]
        return list(map_group[data_type])

    def _get_scalar_map_data(
        self, hdf5_file, dataset_name, sub_id, scalar_map
    ):
        return self._get_data(
            hdf5_file,
            dataset_name,
            sub_id,
            hdf5_dataset_keys.DIFFUSION_GROUP,
            hdf5_dataset_keys.SCALAR_MAP_GROUP,
            scalar_map,
        )

    def _get_label_map_data(self, hdf5_file, dataset_name, sub_id):
        return self._get_data(
            hdf5_file,
            dataset_name,
            sub_id,
            hdf5_dataset_keys.STRUCTURAL_GROUP,
            hdf5_dataset_keys.LABEL_MAP_GROUP,
            hdf5_dataset_keys.SEGMENTATION_DATA,
        )

    def _get_split_group(self, hdf5_file, dataset_name):
        dataset_group = hdf5_file[dataset_name]
        sub_group = dataset_group[hdf5_dataset_keys.SUBJECT_GROUP]
        return sub_group[self._learning_split.value]

    def _get_subject_ids(self, hdf5_file, dataset_name):
        learning_split = self._get_split_group(hdf5_file, dataset_name)
        return list(learning_split.keys())

    def get_subject_ids(self):
        return self._sub_ids
