# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from dmriseg.data.lut.utils import (
    SuitAtlasDiedrichsenGroups,
    get_diedrichsen_group_labels,
)
from dmriseg.utils.contrast_utils import get_contrast_names_lut

contrast_label = "contrast"
subject_label = "subject"


def prepare_data_for_anova(dfs, measure, contrast_names):

    contrast_names_lut = get_contrast_names_lut()

    suit_labels = get_diedrichsen_group_labels(
        SuitAtlasDiedrichsenGroups.ALL.value
    )

    # Compute the mean across all labels for each participant/contrast
    columns_of_interest = list(map(str, suit_labels))

    measure_prtcpnt_mean = np.hstack(
        [df[columns_of_interest].mean(axis=1).values for df in dfs]
    )

    # Create the values for the participant (subject for AnovaRM) and contrast
    # (within for AnovaRM) columns
    participant_ids = np.hstack([df.index.to_numpy() for df in dfs])
    contrast = np.hstack(
        [
            len(df.index) * [contrast_names_lut[contrast_name]]
            for df, contrast_name in zip(dfs, contrast_names)
        ]
    )

    depvar_label = measure
    _subject_label = subject_label
    within_label = [contrast_label]

    df_anova = pd.DataFrame(
        {
            _subject_label: participant_ids,
            contrast_label: contrast,
            depvar_label: measure_prtcpnt_mean,
        }
    )

    return df_anova, depvar_label, _subject_label, within_label
