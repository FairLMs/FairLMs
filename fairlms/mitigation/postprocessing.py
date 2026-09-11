"""Post-processing mitigators: black-box, fitted on outputs.

These touch no model. They fit a decision rule on scores that have already been
produced, which is why they apply to every architecture and need only black-box
access. All three return a **serializable** rule, so the fitted object can be
stored, reviewed and applied later without the model that produced the scores.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fairlms.diagnostics.evidence import LabeledScoredGroups
from fairlms.mitigation.base import MitigationResult, Mitigator
from fairlms.mitigation.evidence import CandidateSets

__all__ = [
    "FairnessAwareReranking",
    "GroupAwareThresholding",
    "ScoreCalibration",
]

_ALL_ARCHITECTURES = ("encoder_only", "decoder_only", "encoder_decoder")


def _require_rows(group: str, n: int, minimum: int, mitigator: str) -> None:
    """Refuse a group too small to fit against, rather than quietly skipping it."""
    if n < minimum:
        raise ValueError(
            f"{mitigator}: group {group!r} has {n} row(s), fewer than the {minimum} "
            f"needed to fit against. Supply more rows for this group or drop it "
            f"from the evidence deliberately; it will not be skipped silently."
        )


def _require_both_classes(
    group: str, positives: Sequence[bool], mitigator: str
) -> None:
    n_pos = sum(positives)
    if n_pos == 0 or n_pos == len(positives):
        raise ValueError(
            f"{mitigator}: group {group!r} has only one outcome class "
            f"({n_pos} positive of {len(positives)}). A per-group rule cannot be "
            f"fitted against a constant outcome."
        )


# ---------------------------------------------------------------------------
# Platt / isotonic calibration
# ---------------------------------------------------------------------------
def _fit_platt(scores: Sequence[float], positives: Sequence[bool]) -> Dict[str, float]:
    """Fit a 1-D logistic ``sigmoid(a * s + b)`` by Newton-Raphson.

    Implemented directly rather than via scikit-learn so that calibration works
    in the base install. Newton on a 2-parameter convex problem converges in a
    handful of iterations and needs no optimizer dependency.
    """
    y = [1.0 if p else 0.0 for p in positives]
    a, b = 0.0, 0.0
    for _ in range(100):
        g_a = g_b = h_aa = h_ab = h_bb = 0.0
        for s, target in zip(scores, y):
            p = 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, a * s + b))))
            residual = p - target
            w = p * (1.0 - p)
            g_a += residual * s
            g_b += residual
            h_aa += w * s * s
            h_ab += w * s
            h_bb += w
        # Ridge term keeps the Hessian invertible when a group is separable.
        h_aa += 1e-9
        h_bb += 1e-9
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-15:
            break
        step_a = (h_bb * g_a - h_ab * g_b) / det
        step_b = (h_aa * g_b - h_ab * g_a) / det
        a -= step_a
        b -= step_b
        if max(abs(step_a), abs(step_b)) < 1e-10:
            break
    return {"a": a, "b": b}


def _fit_isotonic(
    scores: Sequence[float], positives: Sequence[bool]
) -> Dict[str, List[float]]:
    """Fit a monotone step function by pool-adjacent-violators (PAVA)."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    xs = [float(scores[i]) for i in order]
    ys = [1.0 if positives[i] else 0.0 for i in order]

    # Each block is [sum_y, weight]; merge while the sequence decreases.
    values: List[float] = []
    weights: List[float] = []
    for y in ys:
        values.append(y)
        weights.append(1.0)
        while len(values) > 1 and values[-2] > values[-1]:
            v2, w2 = values.pop(), weights.pop()
            v1, w1 = values.pop(), weights.pop()
            values.append((v1 * w1 + v2 * w2) / (w1 + w2))
            weights.append(w1 + w2)

    fitted: List[float] = []
    for value, weight in zip(values, weights):
        fitted.extend([value] * int(weight))
    return {"x": xs, "y": fitted}


def _apply_platt(rule: Dict[str, float], score: float) -> float:
    z = max(-500.0, min(500.0, rule["a"] * score + rule["b"]))
    return 1.0 / (1.0 + math.exp(-z))


def _apply_isotonic(rule: Dict[str, List[float]], score: float) -> float:
    xs, ys = rule["x"], rule["y"]
    if score <= xs[0]:
        return ys[0]
    if score >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] < score:
            lo = mid + 1
        else:
            hi = mid
    return ys[lo]


class ScoreCalibration(Mitigator):
    """Platt or isotonic calibration fitted **per group**.

    A single global calibrator leaves each group's scores miscalibrated in a
    different direction; fitting per group is what makes a downstream threshold
    mean the same thing for everyone.

    Parameters
    ----------
    method:
        ``"platt"`` for a logistic fit, ``"isotonic"`` for a monotone step fit.
    min_rows_per_group:
        Refuse any group with fewer rows than this. Never skip it silently.

    Examples
    --------
    >>> from fairlms.diagnostics import LabeledScoredGroups, ScoredGroups
    >>> from fairlms.mitigation import ScoreCalibration
    >>> evidence = LabeledScoredGroups(
    ...     scored=ScoredGroups(
    ...         axis="gender",
    ...         groups=["f", "f", "f", "m", "m", "m"],
    ...         scores=[0.2, 0.6, 0.9, 0.1, 0.5, 0.8],
    ...         score_name="p_hire", source="doctest", score_range=[0.0, 1.0],
    ...     ),
    ...     labels=["no", "yes", "yes", "no", "no", "yes"],
    ...     label_name="outcome", positive_label="yes",
    ... )
    >>> outcome = ScoreCalibration().apply(None, evidence)
    >>> outcome.category
    'post'
    >>> tuple(sorted(outcome.result["groups"]))
    ('f', 'm')
    """

    name = "score_calibration"
    category = "post"
    access = "black_box"
    architectures = _ALL_ARCHITECTURES
    requires = frozenset()  # operates on scores, not on a model
    accepts = (LabeledScoredGroups,)

    def __init__(self, method: str = "platt", min_rows_per_group: int = 3):
        self.method = method
        self.min_rows_per_group = min_rows_per_group

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        if self.method not in ("platt", "isotonic"):
            raise ValueError(
                f"method must be 'platt' or 'isotonic'; got {self.method!r}."
            )
        fitted = {}
        for group, scores, positives in evidence.iter_groups():
            _require_rows(group, len(scores), self.min_rows_per_group, self.name)
            _require_both_classes(group, positives, self.name)
            fitted[group] = (
                _fit_platt(scores, positives)
                if self.method == "platt"
                else _fit_isotonic(scores, positives)
            )
        return self._result(
            {"method": self.method, "groups": fitted},
            axis=evidence.axis,
            n_rows=evidence.n_rows,
            positive_label=evidence.positive_label,
            groups=list(evidence.support),
        )

    @staticmethod
    def transform(rule: Dict[str, Any], group: str, score: float) -> float:
        """Apply a fitted rule from :attr:`MitigationResult.result` to one score."""
        if group not in rule["groups"]:
            raise KeyError(
                f"no calibrator was fitted for group {group!r}; fitted groups are "
                f"{sorted(rule['groups'])}."
            )
        per_group = rule["groups"][group]
        if rule["method"] == "platt":
            return _apply_platt(per_group, score)
        return _apply_isotonic(per_group, score)


# ---------------------------------------------------------------------------
# Group-aware thresholding
# ---------------------------------------------------------------------------
def _rates(scores: Sequence[float], positives: Sequence[bool], threshold: float):
    """Return ``(tpr, fpr)`` at *threshold*, predicting positive when ``s >= t``."""
    tp = fp = pos = neg = 0
    for score, is_positive in zip(scores, positives):
        predicted = score >= threshold
        if is_positive:
            pos += 1
            tp += predicted
        else:
            neg += 1
            fp += predicted
    return (tp / pos if pos else 0.0), (fp / neg if neg else 0.0)


def _candidate_thresholds(scores: Sequence[float]) -> List[float]:
    """Every threshold that produces a distinct split, plus one above the top."""
    unique = sorted(set(float(s) for s in scores))
    return unique + [unique[-1] + 1.0]


class GroupAwareThresholding(Mitigator):
    """Per-group decision thresholds meeting equal opportunity or equalized odds.

    Searches each group's achievable ``(tpr, fpr)`` operating points and picks
    the per-group thresholds whose rates agree most closely across groups.

    ``equal_opportunity`` matches true-positive rates only; ``equalized_odds``
    matches true-positive and false-positive rates jointly. Which is the right
    criterion is a question about the deployment, so it is a declared parameter
    with no default that pretends otherwise.

    Reports the residual gap it actually achieved: with finite data the rates
    rarely match exactly, and rounding that away would overstate the result.

    **Equalising the rates is not sufficient on its own.** Two operating points
    equalise every rate perfectly and are useless: rejecting everyone (all rates
    zero) and accepting everyone (all rates one). A search that minimises only
    the gap finds one of them and reports a perfect score, because a gap of zero
    is exactly what it was asked for. Among the points that tie on the gap this
    therefore maximises Youden's J, ``mean(TPR) - mean(FPR)``, which is zero for
    both degenerate points and positive for any genuinely discriminating rule.
    The achieved value is reported as ``achieved_utility`` and the rule is
    recorded in provenance, so the trade-off stays visible rather than implied.

    Parameters
    ----------
    criterion:
        ``"equal_opportunity"`` or ``"equalized_odds"``.
    min_rows_per_group:
        Refuse any group with fewer rows than this.

    Examples
    --------
    >>> from fairlms.diagnostics import LabeledScoredGroups, ScoredGroups
    >>> from fairlms.mitigation import GroupAwareThresholding
    >>> evidence = LabeledScoredGroups(
    ...     scored=ScoredGroups(
    ...         axis="gender",
    ...         groups=["f", "f", "f", "f", "m", "m", "m", "m"],
    ...         scores=[0.1, 0.4, 0.6, 0.9, 0.3, 0.5, 0.7, 0.95],
    ...         score_name="p_hire", source="doctest", score_range=[0.0, 1.0],
    ...     ),
    ...     labels=["no", "no", "yes", "yes", "no", "no", "yes", "yes"],
    ...     label_name="outcome", positive_label="yes",
    ... )
    >>> rule = GroupAwareThresholding().apply(None, evidence).result
    >>> tuple(sorted(rule["thresholds"]))
    ('f', 'm')
    >>> rule["achieved_tpr_gap"] <= 1.0
    True
    """

    name = "group_aware_thresholding"
    category = "post"
    access = "black_box"
    architectures = _ALL_ARCHITECTURES
    requires = frozenset()
    accepts = (LabeledScoredGroups,)

    def __init__(
        self,
        criterion: str = "equal_opportunity",
        min_rows_per_group: int = 3,
    ):
        self.criterion = criterion
        self.min_rows_per_group = min_rows_per_group

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        if self.criterion not in ("equal_opportunity", "equalized_odds"):
            raise ValueError(
                "criterion must be 'equal_opportunity' or 'equalized_odds'; got "
                f"{self.criterion!r}."
            )

        # Per group: every achievable (threshold, tpr, fpr) operating point.
        options: Dict[str, List[Tuple[float, float, float]]] = {}
        for group, scores, positives in evidence.iter_groups():
            _require_rows(group, len(scores), self.min_rows_per_group, self.name)
            _require_both_classes(group, positives, self.name)
            options[group] = [
                (t, *_rates(scores, positives, t))
                for t in _candidate_thresholds(scores)
            ]

        groups = sorted(options)
        # Sweep a shared target rate and let each group pick its closest point.
        # Linear in the number of candidate points per group, rather than the
        # exponential cost of searching threshold combinations directly.
        best: Optional[Dict[str, Any]] = None
        targets = sorted({point[1] for pts in options.values() for point in pts})
        for target_tpr in targets:
            chosen = {}
            for group in groups:
                # Closest to the shared target TPR, then the *lowest* FPR among
                # the points that reach it. Several thresholds usually deliver
                # the same TPR; picking by threshold alone would silently take
                # the one that also admits the most false positives.
                chosen[group] = min(
                    options[group],
                    key=lambda p: (abs(p[1] - target_tpr), p[2], p[0]),
                )
            tprs = [chosen[g][1] for g in groups]
            fprs = [chosen[g][2] for g in groups]
            tpr_gap = max(tprs) - min(tprs)
            fpr_gap = max(fprs) - min(fprs)
            cost = tpr_gap + (fpr_gap if self.criterion == "equalized_odds" else 0.0)
            utility = (sum(tprs) - sum(fprs)) / len(groups)
            # Minimise the gap first, then break ties on utility. Rounding keeps
            # float noise from deciding the ordering; the thresholds themselves
            # are the final tie-break so the result is deterministic.
            key = (
                round(cost, 12),
                -round(utility, 12),
                tuple(chosen[g][0] for g in groups),
            )
            if best is None or key < best["key"]:
                best = {
                    "key": key,
                    "thresholds": {g: chosen[g][0] for g in groups},
                    "tpr": {g: chosen[g][1] for g in groups},
                    "fpr": {g: chosen[g][2] for g in groups},
                    "tpr_gap": tpr_gap,
                    "fpr_gap": fpr_gap,
                    "utility": utility,
                }

        assert best is not None  # at least two groups are guaranteed upstream
        return self._result(
            {
                "criterion": self.criterion,
                "thresholds": best["thresholds"],
                "tpr": best["tpr"],
                "fpr": best["fpr"],
                "achieved_tpr_gap": best["tpr_gap"],
                "achieved_fpr_gap": best["fpr_gap"],
                "achieved_utility": best["utility"],
                "positive_label": evidence.positive_label,
            },
            axis=evidence.axis,
            n_rows=evidence.n_rows,
            groups=groups,
            selection_rule=(
                "minimise the rate gap, then maximise Youden's J "
                "(mean TPR - mean FPR) among the operating points that tie"
            ),
        )

    @staticmethod
    def decide(rule: Dict[str, Any], group: str, score: float) -> bool:
        """Apply a fitted threshold rule to one score."""
        if group not in rule["thresholds"]:
            raise KeyError(
                f"no threshold was fitted for group {group!r}; fitted groups are "
                f"{sorted(rule['thresholds'])}."
            )
        return score >= rule["thresholds"][group]


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------
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
