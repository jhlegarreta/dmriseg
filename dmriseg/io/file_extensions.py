# -*- coding: utf-8 -*-

import enum

fname_sep = "."
asterisk = "*"


class LutDataExtension(enum.Enum):
    LUT = "lut"


class TextSepDataExtension(enum.Enum):
    CSV = "csv"
    TSV = "tsv"


class NiftiDataExtension(enum.Enum):
    NII = "nii"


class NrrdDataExtension(enum.Enum):
    NRRD = "nrrd"


class Compression(enum.Enum):
    GZ = "gz"


class PyTorchCheckpointExtension(enum.Enum):
    PT = "pt"
