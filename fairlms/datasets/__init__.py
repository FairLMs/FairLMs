"""Reusable dataset loaders for fairlms metrics."""

from fairlms.datasets.base import FairnessDataset
from fairlms.datasets.bbq import BBQ, DEFAULT_BBQ_CATEGORIES
from fairlms.datasets.bias_in_bios import BIOS_PROFESSION_MAP, BiasInBios
from fairlms.datasets.bias_nli import BIAS_NLI_LABELS, BIAS_NLI_SPLITS, BiasNLI
from fairlms.datasets.bold import (
    BOLD,
    BOLD_DOMAIN_BIAS_TYPES,
    BOLD_DOMAINS,
)
from fairlms.datasets.crows_pairs import CrowSPairs
from fairlms.datasets.eec import EEC, EquityEvaluationCorpus
from fairlms.datasets.gap import GAP, GAP_SPLITS
from fairlms.datasets.grep_biasir import (
    GREP_BIASIR_DOCUMENT_GENDERS,
    GREP_BIASIR_DOMAINS,
    GrepBiasIR,
)
from fairlms.datasets.holistic_bias import HOLISTIC_BIAS_CONFIGS, HolisticBias
from fairlms.datasets.honest import (
    HONEST,
    HONEST_BINARY_LANGUAGES,
    HONEST_CONFIGS,
)
from fairlms.datasets.real_toxicity_prompts import (
    RTP_ATTRIBUTES,
    RealToxicityPrompts,
)
from fairlms.datasets.reddit_bias import (
    REDDIT_BIAS_AXES,
    REDDIT_BIAS_SUBSETS,
    RedditBias,
)
from fairlms.datasets.stereoset import StereoSet
from fairlms.datasets.trustgpt import (
    TRUSTGPT_DEFAULT_ENTITIES,
    TRUSTGPT_TASKS,
    TRUSTGPT_TEMPLATES,
    TrustGPT,
)
from fairlms.datasets.unqover import UNQOVER_MODELS, UNQOVER_SUBJECTS, UnQover
from fairlms.datasets.wino_bias import (
    WINOBIAS_FEMALE_OCC,
    WINOBIAS_MALE_OCC,
    WinoBias,
)
from fairlms.datasets.winogender import WINOGENDER_GENDERS, Winogender
from fairlms.datasets.xnli import XNLIReligionPairs

__all__ = [
    "BBQ",
    "BIAS_NLI_LABELS",
    "BIAS_NLI_SPLITS",
    "BIOS_PROFESSION_MAP",
    "BOLD",
    "BOLD_DOMAINS",
    "BOLD_DOMAIN_BIAS_TYPES",
    "BiasInBios",
    "BiasNLI",
    "CrowSPairs",
    "DEFAULT_BBQ_CATEGORIES",
    "EEC",
    "EquityEvaluationCorpus",
    "FairnessDataset",
    "GAP",
    "GAP_SPLITS",
    "GREP_BIASIR_DOCUMENT_GENDERS",
    "GREP_BIASIR_DOMAINS",
    "GrepBiasIR",
    "HOLISTIC_BIAS_CONFIGS",
    "HONEST",
    "HONEST_BINARY_LANGUAGES",
    "HONEST_CONFIGS",
    "HolisticBias",
    "RTP_ATTRIBUTES",
    "REDDIT_BIAS_AXES",
    "REDDIT_BIAS_SUBSETS",
    "RealToxicityPrompts",
    "RedditBias",
    "StereoSet",
    "TRUSTGPT_DEFAULT_ENTITIES",
    "TRUSTGPT_TASKS",
    "TRUSTGPT_TEMPLATES",
    "TrustGPT",
    "UNQOVER_MODELS",
    "UNQOVER_SUBJECTS",
    "UnQover",
    "WINOBIAS_FEMALE_OCC",
    "WINOBIAS_MALE_OCC",
    "WINOGENDER_GENDERS",
    "WinoBias",
    "Winogender",
    "XNLIReligionPairs",
]
