# -*- coding: utf-8 -*-

import enum

fname_sep = "."
asterisk = "*"


class TextFileExtension(enum.Enum):
    TXT = "txt"


class LutFileExtension(enum.Enum):
    LUT = "lut"


class DelimitedValuesFileExtension(enum.Enum):
    CSV = "csv"
    TSV = "tsv"


class NiftiFileExtension(enum.Enum):
    NII = "nii"


class NrrdFileExtension(enum.Enum):
    NHDR = "nhdr"
    NRRD = "nrrd"
    RAW = "raw"


class CompressedFileExtension(enum.Enum):
    GZ = "gz"


class PyTorchCheckpointFileExtension(enum.Enum):
    PT = "pt"


class FigureFileExtension(enum.Enum):
    PNG = "png"
