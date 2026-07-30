"""Encoder-decoder stereotypical-association metrics: SD, SVA."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from fairllms.definition.encoder_decoder.intrinsic_bias.stereotypical_association.sd.sd import (
    compute_sd,
)
from fairllms.definition.encoder_decoder.intrinsic_bias.stereotypical_association.sva.sva import (
    compute_sva,
)
from fairllms.metrics._compat import take, unwrap, warn_legacy
from fairllms.metrics.base import FairnessMetric, MetricResult
from fairllms.metrics.data import LabeledSentences, StereotypeLabelled, WordSets
from fairllms.metrics.resolve import get_tokenizer_model


class StereotypicalDivergence(FairnessMetric):
    """Stereotype vs anti-stereotype task-performance divergence (SD).

    ``data`` is a :class:`~fairllms.metrics.data.StereotypeLabelled` pairing two
    labelled sentence sets.

    Parameters
    ----------
    max_new_tokens:
        Generation budget per sentence.
    metric_fn:
        Optional callable scoring a prediction against its label. ``None`` uses
        the built-in exact-match comparison.
    """

    name = "stereotypical_divergence"
    bias_type = "intrinsic"
    architectures = ("encoder_decoder",)

    def __init__(
        self, *, max_new_tokens: int = 128, metric_fn: Optional[Callable] = None
    ):
        self.max_new_tokens = max_new_tokens
        self.metric_fn = metric_fn

    def compute(
        self,
        model: Any = None,
        data: Any = None,
        *,
        tokenizer: Any = None,
        **legacy: Any,
    ) -> MetricResult:
        data = unwrap(
            data if data is not None else legacy.pop("dataset", None), StereotypeLabelled
        )

        if data is None:
            ss, k1 = take(legacy, "stereo_sentences")
            sl, k2 = take(legacy, "stereo_labels")
            asents, k3 = take(legacy, "anti_sentences")
            al, k4 = take(legacy, "anti_labels")
            if None in (ss, sl, asents, al):
                raise ValueError(
                    "StereotypicalDivergence requires two labelled sentence sets. "
                    "Pass a StereotypeLabelled as the second argument, e.g. "
                    "compute(model, StereotypeLabelled("
                    "LabeledSentences(stereo, stereo_labels), "
                    "LabeledSentences(anti, anti_labels)))."
                )
            warn_legacy("StereotypicalDivergence", [k1, k2, k3, k4], "StereotypeLabelled")
            for key in (k1, k2, k3, k4):
                legacy.pop(key, None)
            data = StereotypeLabelled(
                LabeledSentences(ss, sl), LabeledSentences(asents, al)
            )

        self._reject_unknown_kwargs(legacy, "max_new_tokens", "metric_fn")
        if not isinstance(data, StereotypeLabelled):
            raise TypeError(
                f"StereotypicalDivergence expects a StereotypeLabelled as data, got "
                f"{type(data).__name__}."
            )

        metric_fn = legacy.get("metric_fn", self.metric_fn)
        sd_kwargs = {"max_new_tokens": legacy.get("max_new_tokens", self.max_new_tokens)}
        if metric_fn is not None:
            sd_kwargs["metric_fn"] = metric_fn

        tok, hf_model, _ = get_tokenizer_model(model, tokenizer)
        m_stereo, m_anti, delta_s, rows = compute_sd(
            hf_model,
            tok,
            list(data.stereotype.sentences),
            list(data.stereotype.labels),
            list(data.anti_stereotype.sentences),
            list(data.anti_stereotype.labels),
            **sd_kwargs,
        )
        return MetricResult(
            score=float(delta_s),
            details={
                "m_stereo": m_stereo,
                "m_anti": m_anti,
                "delta_s": float(delta_s),
                "rows": rows,
            },
        )


class StereotypicalValueAttribution(FairnessMetric):
    """Shapley attribution of stereotype bias to attention heads (SVA).

    ``data`` is a :class:`~fairllms.metrics.data.WordSets` whose ``target_1`` /
    ``target_2`` hold the stereotypical and anti-stereotypical sentences.
    Attribute roles are unused, so pass the same sentences again if you have no
    attribute sets.

    The stereotype ``direction`` is derived from the sentences when not supplied,
    which also guarantees it matches the model's hidden size — previously the
    caller had to supply a correctly-sized vector by hand.

    Parameters
    ----------
    n_samples:
        Monte-Carlo permutations for the Shapley estimate.
    top_pct:
        Fraction of heads whose attribution mass is reported as the score.
    direction:
        Optional precomputed unit direction of length ``d_model``.
    n_layers, n_heads:
        Optional overrides; derived from ``model.config`` when omitted.
    """

    name = "stereotypical_value_attribution"
    bias_type = "intrinsic"
    architectures = ("encoder_decoder",)

    def __init__(
        self,
        *,
        n_samples: int = 15,
        top_pct: float = 0.10,
        direction: Any = None,
        n_layers: Optional[int] = None,
        n_heads: Optional[int] = None,
    ):
        self.n_samples = n_samples
        self.top_pct = top_pct
        self.direction = direction
        self.n_layers = n_layers
        self.n_heads = n_heads

    @staticmethod
    def _derive_shape(config, n_layers, n_heads):
        if n_layers is None:
            n_layers = getattr(config, "num_layers", None) or getattr(
                config, "num_hidden_layers", None
            )
        if n_heads is None:
            n_heads = getattr(config, "num_heads", None) or getattr(
                config, "num_attention_heads", None
            )
        if n_layers is None or n_heads is None:
            raise ValueError(
                "StereotypicalValueAttribution could not derive n_layers/n_heads "
                "from model.config; pass them to the constructor."
            )
        return int(n_layers), int(n_heads)

    def compute(
        self,
        model: Any = None,
        data: Any = None,
        *,
        tokenizer: Any = None,
        **legacy: Any,
    ) -> MetricResult:
        data = unwrap(data if data is not None else legacy.pop("dataset", None), WordSets)

        if data is None:
            stereo, k1 = take(legacy, "stereo_sents")
            anti, k2 = take(legacy, "anti_sents")
            if None in (stereo, anti):
                raise ValueError(
                    "StereotypicalValueAttribution requires stereotypical and "
                    "anti-stereotypical sentences. Pass a WordSets as the second "
                    "argument, e.g. compute(model, WordSets(stereo, anti, stereo, anti))."
                )
            warn_legacy("StereotypicalValueAttribution", [k1, k2], "WordSets")
            legacy.pop(k1, None)
            legacy.pop(k2, None)
            data = WordSets(stereo, anti, stereo, anti)

        self._reject_unknown_kwargs(
            legacy, "n_samples", "top_pct", "direction", "n_layers", "n_heads"
        )
        if not isinstance(data, WordSets):
            raise TypeError(
                f"StereotypicalValueAttribution expects a WordSets as data, got "
                f"{type(data).__name__}."
            )

        tok, hf_model, _ = get_tokenizer_model(model, tokenizer)
        n_layers, n_heads = self._derive_shape(
            hf_model.config,
            legacy.get("n_layers", self.n_layers),
            legacy.get("n_heads", self.n_heads),
        )

        stereo_sents = list(data.target_1)
        anti_sents = list(data.target_2)

        direction = legacy.get("direction", self.direction)
        if direction is None:
            from fairllms.definition.encoder_decoder.intrinsic_bias.stereotypical_association.sva.sva import (  # noqa: E501
                compute_stereotype_direction,
            )

            direction = compute_stereotype_direction(
                hf_model, tok, stereo_sents, anti_sents
            )
            derived_direction = True
        else:
            direction = np.asarray(direction, dtype=float)
            derived_direction = False
            expected = getattr(hf_model.config, "d_model", None) or getattr(
                hf_model.config, "hidden_size", None
            )
            if expected is not None and direction.shape[-1] != expected:
                raise ValueError(
                    f"direction has length {direction.shape[-1]} but the model's "
                    f"hidden size is {expected}. Omit direction= to derive it from "
                    f"the sentences."
                )

        sva, phi = compute_sva(
            hf_model,
            tok,
            stereo_sents,
            anti_sents,
            np.asarray(direction),
            n_layers,
            n_heads,
            n_samples=legacy.get("n_samples", self.n_samples),
            top_pct=legacy.get("top_pct", self.top_pct),
        )
        return MetricResult(
            score=float(sva),
            details={
                "phi": phi,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "direction_derived": derived_direction,
            },
        )
