# -*- coding: utf-8 -*-

import enum

import numpy as np
import pandas as pd

mode_label = "mode"
count_label = "count"


class StatisticalTest(enum.Enum):
    PAIRWISE_T_TEST = "pairwise_t_test"
    WILCOXON_RANKSUM = "wilcoxon_ranksum"


def mode_count(series):

    mode_val = series.mode()
    if len(mode_val) > 0:
        mode = mode_val[0]
        count = (series == mode).sum()
    else:
        mode = np.nan
        count = 0
    return pd.Series([mode, count], index=[mode_label, count_label])
