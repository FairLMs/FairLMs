"""StereoSet dataset loader."""

from __future__ import annotations

from typing import List, Optional, Sequence

from fairlms.datasets.base import FairnessDataset, optional_limit

# Official StereoSet gold_label convention (Hugging Face / McGill-NLP):
#   0 = stereotype, 1 = anti-stereotype, 2 = unrelated
_OFFICIAL_LABEL = {0: "stereo", 1: "anti", 2: "unrelated"}


class StereoSet(FairnessDataset):
    """Load StereoSet as pairs or triples.

    Parameters
    ----------
    config:
        ``\"intersentence\"`` (default) or ``\"intrasentence\"``.
    as_triples:
        If True, require stereotype / anti_stereotype / unrelated.
        If False, return stereotype / anti_stereotype pairs only.
    hf_path:
        Hugging Face dataset id. Defaults try both common ids.
    label_map:
        Optional override of gold_label → role mapping. Defaults to the
        official StereoSet mapping ``{0: stereo, 1: anti, 2: unrelated}``.

    Each pair example::

        {"stereotype", "anti_stereotype", "bias_type"}

    Each triple example::

        {"stereotype", "anti_stereotype", "unrelated", "bias_type"}
    """

    name = "stereoset"

    def __init__(
        self,
        config: str = "intersentence",
        split: str = "validation",
        as_triples: bool = False,
        n_max: Optional[int] = None,
        hf_path: Optional[str] = None,
        label_map: Optional[dict] = None,
    ):
        self.config = config
        self.split = split
        self.as_triples = as_triples
        self.n_max = n_max
        self.hf_path = hf_path
        self.label_map = label_map or dict(_OFFICIAL_LABEL)
        self._cache: Optional[List[dict]] = None

    def _load_hf(self):
        from datasets import load_dataset

        errors = []
        candidates = (
            [self.hf_path]
            if self.hf_path
            else ["stereoset", "McGill-NLP/stereoset"]
        )
        for path in candidates:
            if not path:
                continue
            try:
                return load_dataset(path, self.config, split=self.split)
            except Exception as exc:  # noqa: BLE001 - try next candidate
                errors.append(f"{path}: {exc}")
        raise RuntimeError(
            "Failed to load StereoSet from Hugging Face. Tried: "
            + "; ".join(errors)
        )

    def load(self) -> Sequence[dict]:
        if self._cache is not None:
            return optional_limit(self._cache, self.n_max)

        ds = self._load_hf()
        examples: List[dict] = []

        for row in ds:
            bt = row.get("bias_type")
            bucket = {"stereo": None, "anti": None, "unrelated": None}
            for sent, label in zip(
                row["sentences"]["sentence"], row["sentences"]["gold_label"]
            ):
                role = self.label_map.get(label)
                if role in bucket and bucket[role] is None:
                    bucket[role] = sent

            if self.as_triples:
                if bucket["stereo"] and bucket["anti"] and bucket["unrelated"]:
                    examples.append(
                        {
                            "stereotype": bucket["stereo"],
                            "anti_stereotype": bucket["anti"],
                            "unrelated": bucket["unrelated"],
                            "bias_type": bt,
                        }
                    )
            elif bucket["stereo"] and bucket["anti"]:
                examples.append(
                    {
                        "stereotype": bucket["stereo"],
                        "anti_stereotype": bucket["anti"],
                        "bias_type": bt,
                    }
                )

            if self.n_max is not None and len(examples) >= self.n_max:
                break

        self._cache = examples
        return optional_limit(examples, self.n_max)
