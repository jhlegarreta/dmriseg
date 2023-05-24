# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader

from dmriseg.dataset.dmri_dataset import DMRISegDataset
from dmriseg.io.learning_description import LearningSplit


class DMRIDataManager:
    def __init__(self, dataset_fname):

        self._dataset_fname = dataset_fname

    def get_loader(self, learning_split, **kwargs):

        dataset = DMRISegDataset(self._dataset_fname, learning_split)

        shuffle = get_shuffle(learning_split)

        return DataLoader(dataset, shuffle=shuffle, **kwargs)


def get_shuffle(split):

    if split == LearningSplit.TRAIN or split == LearningSplit.VALID:
        return True
    else:
        return False
