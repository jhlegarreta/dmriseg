# -*- coding: utf-8 -*-

import enum


class ContrastNames(enum.Enum):
    T1 = "t1"
    B0 = "b0"
    DWI = "dwi"
    DWI1k = "dwi1k"
    DWI2k = "dwi2k"
    DWI3k = "dwi3k"
    FA = "fa"
    MD = "md"
    RD = "rd"
    EVALS_E1 = "evalse1"
    EVALS_E2 = "evalse2"
    EVALS_E3 = "evalse3"
    AK = "ak"
    MK = "mk"
    RK = "rk"


def get_contrast_names_lut():
    return dict(
        {
            ContrastNames.T1.value: 1,
            ContrastNames.B0.value: 2,
            ContrastNames.DWI.value: 3,
            ContrastNames.DWI1k.value: 4,
            ContrastNames.DWI2k.value: 5,
            ContrastNames.DWI3k.value: 6,
            ContrastNames.FA.value: 7,
            ContrastNames.MD.value: 8,
            ContrastNames.RD.value: 9,
            ContrastNames.EVALS_E1.value: 10,
            ContrastNames.EVALS_E2.value: 11,
            ContrastNames.EVALS_E3.value: 12,
            ContrastNames.AK.value: 13,
            ContrastNames.MK.value: 14,
            ContrastNames.RK.value: 15,
        }
    )
