# -*- coding: utf-8 -*-

import enum


class LearningData(enum.Enum):
    SPLIT = "split"


class LearningSplit(enum.Enum):
    TEST = "test"
    TRAIN = "train"
    VALID = "valid"
