"""Fairness-aware reranking: reorder candidates by a declared bias scorer."""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

from fairlms.mitigation.base import MitigationResult, Mitigator
from fairlms.mitigation.evidence import CandidateSets

__all__ = ["FairnessAwareReranking"]


def _rerank_one(
    query: Any,
    candidates: Sequence[Any],
    *,
    scorer: Any,
    weight: float,
    owner: str,
) -> Tuple[list, list]:
    """Reorder one candidate list; the single definition of the ordering.

    Shared by :meth:`FairnessAwareReranking._apply` and
    :meth:`FairnessAwareReranking.rerank` so a fitted rule cannot drift from the
    ordering that produced it.
    """
    n = len(candidates)
    scored = []
    for rank, candidate in enumerate(candidates):
        bias = scorer(query, candidate)
        if isinstance(bias, bool) or not isinstance(bias, (int, float)):
            raise TypeError(
                f"{owner}: scorer must return a real number for "
                f"(query={query!r}, candidate={candidate!r}); got "
                f"{type(bias).__name__}."
            )
        bias = float(bias)
        if not math.isfinite(bias):
            raise ValueError(
                f"{owner}: scorer returned a non-finite score for "
                f"(query={query!r}, candidate={candidate!r})."
            )
        # Original position as a descending score in [0, 1].
        original = 1.0 - (rank / (n - 1)) if n > 1 else 1.0
        combined = (1.0 - weight) * original - weight * bias
        scored.append((combined, rank, candidate, bias))
    # Ties keep the generator's original order.
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored], [item[3] for item in scored]


class FairnessAwareReranking(Mitigator):
    """Reorder candidate generations by a **declared** bias scorer.

    The scorer arrives on the evidence, never from the library. Reranking trades
    the generator's own ordering for one the caller is willing to defend, and a
    built-in default would hide that choice.

    Scores are combined as ``(1 - weight) * original_rank_score - weight *
    bias``, so ``weight=0`` reproduces the original order exactly and
    ``weight=1`` ranks purely by the declared bias scorer.

    Parameters
    ----------
    weight:
        How far to move from the original ordering, in ``[0, 1]``.

    Examples
    --------
    >>> from fairlms.mitigation import CandidateSets, FairnessAwareReranking
    >>> evidence = CandidateSets(
    ...     queries=["the nurse said"],
    ...     candidates=[["she smiled", "they smiled"]],
    ...     scorer=lambda q, c: 1.0 if "she" in c else 0.0,
    ...     scorer_name="doctest-gendered-pronoun",
    ... )
    >>> outcome = FairnessAwareReranking(weight=1.0).apply(None, evidence)
    >>> ranking, = outcome.result["rankings"]
    >>> tuple(ranking)
    ('they smiled', 'she smiled')
    """

    name = "fairness_aware_reranking"
    category = "post"
    access = "black_box"
    # Generative-only, and that restriction lives here rather than in
    # `requires`: reranking reorders candidates the caller already generated and
    # reads nothing from a model, so declaring a capability would falsely demand
    # one be supplied.
    architectures = ("decoder_only", "encoder_decoder")
    requires = frozenset()
    accepts = (CandidateSets,)

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"weight must be in [0, 1]; got {self.weight!r}.")

        rankings, bias_scores = [], []
        for query, candidates in zip(evidence.queries, evidence.candidates):
            order, biases = _rerank_one(
                query,
                candidates,
                scorer=evidence.scorer,
                weight=self.weight,
                owner=self.name,
            )
            rankings.append(order)
            bias_scores.append(biases)

        return self._result(
            {
                "rankings": rankings,
                "bias_scores": bias_scores,
                "scorer_name": evidence.scorer_name,
                "weight": self.weight,
            },
            n_queries=evidence.n_queries,
            scorer_name=evidence.scorer_name,
        )

    @staticmethod
    def rerank(
        rule: Dict[str, Any],
        query: Any,
        candidates: Sequence[Any],
        *,
        scorer: Any,
    ) -> Tuple[list, list]:
        """Apply a fitted rule from :attr:`MitigationResult.result` to new candidates.

        Mirrors :meth:`ScoreCalibration.transform` and
        :meth:`GroupAwareThresholding.decide`: the fitted object is a rule, and
        this re-applies it. Returns ``(reordered_candidates, bias_scores)``.

        The scorer is passed in rather than stored, because a scorer is a live
        callable and a rule has to stay serializable. ``rule['scorer_name']``
        records which scorer produced the fit, so the caller can check they are
        re-applying the same one; a mismatch is refused rather than silently
        reordering by a different notion of bias.
        """
        for key in ("weight", "scorer_name"):
            if key not in rule:
                raise KeyError(
                    f"rule is missing {key!r}; pass MitigationResult.result from "
                    "FairnessAwareReranking."
                )
        declared = getattr(scorer, "__name__", None) or type(scorer).__name__
        if rule["scorer_name"] not in (declared, "<lambda>", None):
            if declared != "<lambda>":
                raise ValueError(
                    f"rule was fitted with scorer {rule['scorer_name']!r} but "
                    f"{declared!r} was supplied; re-applying a rule under a "
                    "different scorer would reorder by a different notion of bias."
                )
        return _rerank_one(
            query,
            candidates,
            scorer=scorer,
            weight=float(rule["weight"]),
            owner="FairnessAwareReranking.rerank",
        )
