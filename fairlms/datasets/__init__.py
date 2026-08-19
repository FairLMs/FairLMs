"""Reusable dataset loaders for fairlms metrics."""

from fairlms.datasets.base import FairnessDataset
from fairlms.datasets.bbq import BBQ, DEFAULT_BBQ_CATEGORIES
from fairlms.datasets.bias_in_bios import BIOS_PROFESSION_MAP, BiasInBios
from fairlms.datasets.crows_pairs import CrowSPairs
from fairlms.datasets.stereoset import StereoSet
from fairlms.datasets.wino_bias import (
    WINOBIAS_FEMALE_OCC,
    WINOBIAS_MALE_OCC,
    WinoBias,
)
from fairlms.datasets.xnli import XNLIReligionPairs

__all__ = [
    "BBQ",
    "BIOS_PROFESSION_MAP",
    "BiasInBios",
    "CrowSPairs",
    "DEFAULT_BBQ_CATEGORIES",
    "FairnessDataset",
    "StereoSet",
    "WINOBIAS_FEMALE_OCC",
    "WINOBIAS_MALE_OCC",
    "WinoBias",
    "XNLIReligionPairs",
]
