"""Decoder-only metric wrappers."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from fairLLMs.definition.decoder_only.extrinsic_bias.counterfactual_fairness.cr.cr import (
    compute_cr,
)
from fairLLMs.definition.decoder_only.extrinsic_bias.counterfactual_fairness.ctf.ctf import (
    compute_ctf,
)
from fairLLMs.definition.decoder_only.extrinsic_bias.demographic_representation.dnp.dnp import (
    compute_dnp,
)
from fairLLMs.definition.decoder_only.extrinsic_bias.demographic_representation.drd.drd import (
    compute_drd,
)
from fairLLMs.definition.decoder_only.extrinsic_bias.performance_disparity.ad.ad import (
    compute_ad,
)
from fairLLMs.definition.decoder_only.extrinsic_bias.performance_disparity.ba.ba import (
    compute_ba,
)
from fairLLMs.definition.decoder_only.extrinsic_bias.performance_disparity.sns.sns import (
    compute_sns,
)
from fairLLMs.definition.decoder_only.intrinsic_bias.attention_head_based_disparity.gbe.gbe import (
    compute_gbe,
    compute_gbe_matrix,
)
from fairLLMs.definition.decoder_only.intrinsic_bias.attention_head_based_disparity.nie.nie import (
    compute_nie,
    compute_nie_matrix,
)
from fairLLMs.definition.decoder_only.intrinsic_bias.stereotypical_association.ca.ca import (
    compute_ca,
)
from fairLLMs.definition.decoder_only.intrinsic_bias.stereotypical_association.sll.sll import (
    compute_sll,
)
from fairLLMs.metrics.base import FairnessMetric, MetricResult
from fairLLMs.metrics.resolve import (
    get_examples,
    get_openai_bundle,
    get_tokenizer_model,
    require_kwargs,
)


def _prompt_pairs(dataset, kwargs):
    factual = kwargs.get("factual_prompts")
    counterfactual = kwargs.get("counterfactual_prompts")
    if dataset is not None and (factual is None or counterfactual is None):
        examples = get_examples(dataset) or []
        factual, counterfactual = [], []
        for ex in examples:
            if isinstance(ex, dict):
                factual.append(ex.get("factual") or ex.get("stereotype"))
                counterfactual.append(
                    ex.get("counterfactual") or ex.get("anti_stereotype")
                )
            else:
                factual.append(ex[0])
                counterfactual.append(ex[1])
    return factual, counterfactual


class CounterfactualRobustness(FairnessMetric):
    """Change rate of top-1 token under counterfactual flip (CR)."""

    name = "counterfactual_robustness"
    bias_type = "extrinsic"
    architectures = ("decoder_only",)

    def __init__(self, completion_model: str = "gpt-3.5-turbo-instruct"):
        self.completion_model = completion_model

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        bundle = get_openai_bundle(model)
        factual, counterfactual = _prompt_pairs(dataset, kwargs)
        require_kwargs(
            {"factual_prompts": factual, "counterfactual_prompts": counterfactual},
            "factual_prompts",
            "counterfactual_prompts",
        )
        cr, rows = compute_cr(
            bundle.client,
            factual,
            counterfactual,
            model=kwargs.get("completion_model", self.completion_model),
        )
        return MetricResult(score=float(cr), details={"rows": rows})


class CounterfactualFairnessScore(FairnessMetric):
    """Mean TVD of next-token distributions (CTF)."""

    name = "counterfactual_fairness"
    bias_type = "extrinsic"
    architectures = ("decoder_only",)

    def __init__(self, completion_model: str = "gpt-3.5-turbo-instruct"):
        self.completion_model = completion_model

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        bundle = get_openai_bundle(model)
        factual, counterfactual = _prompt_pairs(dataset, kwargs)
        require_kwargs(
            {"factual_prompts": factual, "counterfactual_prompts": counterfactual},
            "factual_prompts",
            "counterfactual_prompts",
        )
        ctf, rows = compute_ctf(
            bundle.client,
            factual,
            counterfactual,
            model=kwargs.get("completion_model", self.completion_model),
        )
        return MetricResult(score=float(ctf), details={"rows": rows})


class DemographicNextTokenProportion(FairnessMetric):
    """Demographic normalized next-token probabilities (DNP)."""

    name = "demographic_next_token_proportion"
    bias_type = "extrinsic"
    architectures = ("decoder_only",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        tokenizer, hf_model, _ = get_tokenizer_model(model, kwargs.get("tokenizer"))
        prompts = kwargs.get("prompts")
        if prompts is None and dataset is not None:
            examples = get_examples(dataset) or []
            prompts = [
                ex["prompt"] if isinstance(ex, dict) else ex for ex in examples
            ]
        require_kwargs(
            {
                "prompts": prompts,
                "stereo_words": kwargs.get("stereo_words"),
                "counter_words": kwargs.get("counter_words"),
                "neutral_words": kwargs.get("neutral_words"),
            },
            "prompts",
            "stereo_words",
            "counter_words",
            "neutral_words",
        )
        mean_ps, mean_psp, mean_pd, rows = compute_dnp(
            hf_model,
            tokenizer,
            prompts,
            kwargs["stereo_words"],
            kwargs["counter_words"],
            kwargs["neutral_words"],
        )
        return MetricResult(
            score=float(mean_pd),
            details={
                "mean_ps": mean_ps,
                "mean_psp": mean_psp,
                "mean_pd": mean_pd,
                "rows": rows,
            },
        )


class DemographicRepresentationDivergence(FairnessMetric):
    """Demographic representation disparity in generations (DRD)."""

    name = "demographic_representation_divergence"
    bias_type = "extrinsic"
    architectures = ("decoder_only",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        tokenizer, hf_model, _ = get_tokenizer_model(model, kwargs.get("tokenizer"))
        prompts = kwargs.get("prompts")
        if prompts is None and dataset is not None:
            examples = get_examples(dataset) or []
            prompts = [
                ex["prompt"] if isinstance(ex, dict) else ex for ex in examples
            ]
        require_kwargs(
            {
                "prompts": prompts,
                "stereo_words": kwargs.get("stereo_words"),
                "counter_words": kwargs.get("counter_words"),
            },
            "prompts",
            "stereo_words",
            "counter_words",
        )
        drd, n_s, n_sp, rows = compute_drd(
            hf_model,
            tokenizer,
            prompts,
            kwargs["stereo_words"],
            kwargs["counter_words"],
            max_new_tokens=kwargs.get("max_new_tokens", 50),
        )
        return MetricResult(
            score=float(drd),
            details={"n_s_total": n_s, "n_sp_total": n_sp, "rows": rows},
        )


class AccuracyDisparity(FairnessMetric):
    """Accuracy disparity between stereotype and counter-stereotype scores."""

    name = "accuracy_disparity"
    bias_type = "extrinsic"
    architectures = ("decoder_only",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        scores_s = kwargs.get("scores_s")
        scores_sp = kwargs.get("scores_sp")
        if dataset is not None and (scores_s is None or scores_sp is None):
            examples = get_examples(dataset) or []
            scores_s = [ex["scores_s"] if isinstance(ex, dict) else ex[0] for ex in examples]
            scores_sp = [ex["scores_sp"] if isinstance(ex, dict) else ex[1] for ex in examples]
        require_kwargs(
            {"scores_s": scores_s, "scores_sp": scores_sp}, "scores_s", "scores_sp"
        )
        acc_s, acc_sp, ad = compute_ad(scores_s, scores_sp)
        return MetricResult(
            score=float(ad),
            details={"acc_s": acc_s, "acc_sp": acc_sp},
        )


class BiasAmplifierScore(FairnessMetric):
    """BiasAsker absolute & relative bias (BA)."""

    name = "bias_amplifier"
    bias_type = "extrinsic"
    architectures = ("decoder_only",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        bundle = get_openai_bundle(model)
        require_kwargs(kwargs, "groups", "properties", "ab_template", "rb_template")
        ab, rb, ab_rows, rb_rows = compute_ba(
            bundle.client,
            kwargs["groups"],
            kwargs["properties"],
            kwargs["ab_template"],
            kwargs["rb_template"],
            max_new_tokens=kwargs.get("max_new_tokens", 20),
        )
        return MetricResult(
            score=float(ab),
            details={"ab": ab, "rb": rb, "ab_rows": ab_rows, "rb_rows": rb_rows},
        )


class SensitiveNameSimilarity(FairnessMetric):
    """Sensitive-to-neutral similarity (SNS)."""

    name = "sensitive_name_similarity"
    bias_type = "extrinsic"
    architectures = ("decoder_only",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        call_model: Callable = kwargs.get("call_model") or model
        if call_model is None:
            raise TypeError("Pass call_model=... callable or a compatible model")
        require_kwargs(
            kwargs,
            "queries",
            "neutral_prompt_fn",
            "group_prompt_fn",
            "group_values",
        )
        snsr, snsv, df = compute_sns(
            call_model,
            kwargs["queries"],
            kwargs["neutral_prompt_fn"],
            kwargs["group_prompt_fn"],
            kwargs["group_values"],
            k=kwargs.get("k", 5),
        )
        return MetricResult(
            score=float(snsr),
            details={"snsr": snsr, "snsv": snsv, "table": df},
        )


class GradientBasedBiasEstimation(FairnessMetric):
    """Gradient-based bias on GPT-2 Value heads (GBE)."""

    name = "gradient_based_bias_estimation"
    bias_type = "intrinsic"
    architectures = ("decoder_only",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        if "gbe_matrix" in kwargs and kwargs["gbe_matrix"] is not None:
            matrix = np.asarray(kwargs["gbe_matrix"])
        else:
            tokenizer, hf_model, device = get_tokenizer_model(
                model, kwargs.get("tokenizer")
            )
            require_kwargs(kwargs, "X", "Y", "A", "B")
            matrix = compute_gbe_matrix(
                hf_model,
                tokenizer,
                device,
                kwargs["X"],
                kwargs["Y"],
                kwargs["A"],
                kwargs["B"],
                loss_scale=kwargs.get("loss_scale", 1.0),
                verbose=kwargs.get("verbose", False),
            )
        score = float(compute_gbe(matrix))
        return MetricResult(score=score, details={"gbe_matrix": matrix})


class NaturalIndirectEffect(FairnessMetric):
    """Natural Indirect Effect / attention mediation (NIE)."""

    name = "natural_indirect_effect"
    bias_type = "intrinsic"
    architectures = ("decoder_only",)

    def __init__(self, threshold: float = 0.003):
        self.threshold = threshold

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        if "nie" in kwargs and kwargs["nie"] is not None:
            nie = np.asarray(kwargs["nie"])
        else:
            tokenizer, hf_model, device = get_tokenizer_model(
                model, kwargs.get("tokenizer")
            )
            require_kwargs(kwargs, "probes", "N_LAYERS", "N_HEADS", "HEAD_DIM")
            nie = compute_nie_matrix(
                hf_model,
                tokenizer,
                device,
                kwargs["probes"],
                kwargs["N_LAYERS"],
                kwargs["N_HEADS"],
                kwargs["HEAD_DIM"],
            )
        score = float(compute_nie(nie, threshold=kwargs.get("threshold", self.threshold)))
        return MetricResult(score=score, details={"nie": nie})


class CooccurrenceAssociation(FairnessMetric):
    """Concept association via generation + TV distance (CA)."""

    name = "cooccurrence_association"
    bias_type = "intrinsic"
    architectures = ("decoder_only",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        tokenizer, hf_model, _ = get_tokenizer_model(model, kwargs.get("tokenizer"))
        require_kwargs(kwargs, "concepts", "prompt_template", "group_terms")
        mean_tvd, n_valid, n_skipped = compute_ca(
            hf_model,
            tokenizer,
            kwargs["concepts"],
            kwargs["prompt_template"],
            kwargs["group_terms"],
            n_samples=kwargs.get("n_samples", 20),
        )
        return MetricResult(
            score=float(mean_tvd),
            details={"n_valid": n_valid, "n_skipped": n_skipped},
        )


class StereotypicalLogLikelihood(FairnessMetric):
    """Stereotypical log-likelihood gaps (SLL)."""

    name = "stereotypical_log_likelihood"
    bias_type = "intrinsic"
    architectures = ("decoder_only",)

    def compute(self, model: Any = None, dataset: Any = None, **kwargs: Any) -> MetricResult:
        tokenizer, hf_model, device = get_tokenizer_model(model, kwargs.get("tokenizer"))
        occupation_pairs = kwargs.get("occupation_pairs")
        if occupation_pairs is None and dataset is not None:
            occupation_pairs = get_examples(dataset)
        if not occupation_pairs:
            raise ValueError("occupation_pairs=... is required")
        scores = compute_sll(hf_model, tokenizer, device, occupation_pairs)
        # Primary score: mean of NV/CV/IV magnitudes
        primary = float(np.mean([abs(v) for v in scores.values()]))
        return MetricResult(score=primary, details=dict(scores))
