"""CrowS-Pairs dataset loader."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import pandas as pd

from fairLLMs.datasets.base import FairnessDataset, optional_limit
from fairLLMs.utils.paths import resolve_crows_pairs_csv

PathLike = Union[str, Path]


class CrowSPairs(FairnessDataset):
    """Load CrowS-Pairs as stereotype / anti-stereotype sentence pairs.

    Each example is a dict::

        {
            "stereotype": str,
            "anti_stereotype": str,
            "bias_type": str,
            "stereo_antistereo": str,  # original CrowS label
        }
    """

    name = "crows_pairs"

    def __init__(
        self,
        path: Optional[PathLike] = None,
        bias_type: Optional[str] = None,
        n_max: Optional[int] = None,
    ):
        self.path = path
        self.bias_type = bias_type
        self.n_max = n_max
        self._cache: Optional[List[dict]] = None

    def load(self) -> Sequence[dict]:
        if self._cache is not None:
            return optional_limit(self._cache, self.n_max)

        csv_path = resolve_crows_pairs_csv(self.path)
        df = pd.read_csv(csv_path)
        if self.bias_type is not None:
            df = df[df["bias_type"] == self.bias_type]

        pairs: List[dict] = []
        for _, row in df.iterrows():
            if row["stereo_antistereo"] == "stereo":
                stereo, anti = row["sent_more"], row["sent_less"]
            else:
                stereo, anti = row["sent_less"], row["sent_more"]
            pairs.append(
                {
                    "stereotype": stereo,
                    "anti_stereotype": anti,
                    "bias_type": row.get("bias_type"),
                    "stereo_antistereo": row.get("stereo_antistereo"),
                }
            )

        self._cache = pairs
        return optional_limit(pairs, self.n_max)

    def bias_types(self) -> List[str]:
        """Return sorted bias_type labels present in the full CSV (ignores n_max)."""
        full = CrowSPairs(path=self.path, bias_type=None, n_max=None)
        examples = full.load()
        return sorted(
            {
                ex["bias_type"]
                for ex in examples
                if ex.get("bias_type") is not None and str(ex["bias_type"]) != "nan"
            }
        )
