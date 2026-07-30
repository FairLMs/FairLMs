"""Reusable dataset loaders for fairllms metrics."""

from fairllms.datasets.base import FairnessDataset
from fairllms.datasets.bbq import BBQ, DEFAULT_BBQ_CATEGORIES
from fairllms.datasets.bias_in_bios import BIOS_PROFESSION_MAP, BiasInBios
from fairllms.datasets.crows_pairs import CrowSPairs
from fairllms.datasets.stereoset import StereoSet
from fairllms.datasets.wino_bias import (
    WINOBIAS_FEMALE_OCC,
    WINOBIAS_MALE_OCC,
    WinoBias,
)
from fairllms.datasets.xnli import XNLIReligionPairs

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
