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
    DWIsub20 = "dwisub20"
    DWIsub30 = "dwisub30"
    DWIsub60 = "dwisub60"
    DWI1ksub20 = "dwi1ksub20"
    DWI1ksub30 = "dwi1ksub30"
    DWI1ksub60 = "dwi1ksub60"


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
            ContrastNames.DWIsub20.value: 16,
            ContrastNames.DWIsub30.value: 17,
            ContrastNames.DWIsub60.value: 18,
            ContrastNames.DWI1ksub20.value: 19,
            ContrastNames.DWI1ksub30.value: 20,
            ContrastNames.DWI1ksub60.value: 21,
        }
    )


def get_contrast_from_dir_base(dirname):

    # Build the labels for the i/o dirs/files
    if dirname == "dmri_hcp_t1":
        return ContrastNames.T1.value
    elif dirname == "dmri_hcp_b0":
        return ContrastNames.B0.value
    elif dirname == "dmri_hcp_sphm_b1000-2000-3000":
        return ContrastNames.DWI.value
    elif dirname == "dmri_hcp_sphm_b1000":
        return ContrastNames.DWI1k.value
    elif dirname == "dmri_hcp_sphm_b2000":
        return ContrastNames.DWI2k.value
    elif dirname == "dmri_hcp_sphm_b3000":
        return ContrastNames.DWI3k.value
    elif dirname == "dmri_hcp_fa":
        return ContrastNames.FA.value
    elif dirname == "dmri_hcp_md":
        return ContrastNames.MD.value
    elif dirname == "dmri_hcp_rd":
        return ContrastNames.RD.value
    elif dirname == "dmri_hcp_evals_e1":
        return ContrastNames.EVALS_E1.value
    elif dirname == "dmri_hcp_evals_e2":
        return ContrastNames.EVALS_E2.value
    elif dirname == "dmri_hcp_evals_e3":
        return ContrastNames.EVALS_E3.value
    elif dirname == "dmri_hcp_ak":
        return ContrastNames.AK.value
    elif dirname == "dmri_hcp_mk":
        return ContrastNames.MK.value
    elif dirname == "dmri_hcp_rk":
        return ContrastNames.RK.value
    else:
        raise NotImplementedError(f"Dirname not recognized: {dirname}")


def get_dir_base_from_contrast_name(contrast):

    if contrast == ContrastNames.T1.value:
        return "dmri_hcp_t1"
    elif contrast == ContrastNames.B0.value:
        return "dmri_hcp_b0"
    elif contrast == ContrastNames.DWI.value:
        return "dmri_hcp_sphm_b1000-2000-3000"
    elif contrast == ContrastNames.DWI1k.value:
        return "dmri_hcp_sphm_b1000"
    elif contrast == ContrastNames.DWI2k.value:
        return "dmri_hcp_sphm_b2000"
    elif contrast == ContrastNames.DWI3k.value:
        return "dmri_hcp_sphm_b3000"
    elif contrast == ContrastNames.FA.value:
        return "dmri_hcp_fa"
    elif contrast == ContrastNames.MD.value:
        return "dmri_hcp_md"
    elif contrast == ContrastNames.RD.value:
        return "dmri_hcp_rd"
    elif contrast == ContrastNames.EVALS_E1.value:
        return "dmri_hcp_evals_e1"
    elif contrast == ContrastNames.EVALS_E2.value:
        return "dmri_hcp_evals_e2"
    elif contrast == ContrastNames.EVALS_E3.value:
        return "dmri_hcp_evals_e3"
    elif contrast == ContrastNames.AK.value:
        return "dmri_hcp_ak"
    elif contrast == ContrastNames.MK.value:
        return "dmri_hcp_mk"
    elif contrast == ContrastNames.RK.value:
        return "dmri_hcp_rk"
    else:
        raise NotImplementedError(f"Contrast not recognized: {contrast}")


def rename_contrasts_plot_labels(contrast):

    if contrast == ContrastNames.T1.value:
        return "T1w"
    elif contrast == ContrastNames.B0.value:
        return "b0"
    elif contrast == ContrastNames.DWI.value:
        return "SM"
    elif contrast == ContrastNames.DWI1k.value:
        return "SM1k"
    elif contrast == ContrastNames.DWI2k.value:
        return "SM2k"
    elif contrast == ContrastNames.DWI3k.value:
        return "SM3k"
    elif contrast == ContrastNames.FA.value:
        return "FA"
    elif contrast == ContrastNames.MD.value:
        return "MD"
    elif contrast == ContrastNames.RD.value:
        return "RD"
    elif contrast == ContrastNames.EVALS_E1.value:
        return "E1"
    elif contrast == ContrastNames.EVALS_E2.value:
        return "E2"
    elif contrast == ContrastNames.EVALS_E3.value:
        return "E3"
    elif contrast == ContrastNames.AK.value:
        return "AK"
    elif contrast == ContrastNames.MK.value:
        return "MK"
    elif contrast == ContrastNames.RK.value:
        return "RK"
    else:
        raise NotImplementedError(f"Contrast not recognized: {contrast}")
