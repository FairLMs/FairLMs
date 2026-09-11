"""In-processing mitigators: white-box loss components.

Every mitigator here returns a **callable loss component, not a trained model**.
There is no trainer in the core package: the caller owns the optimizer, the
schedule and the data loader, and composes the returned term into their own
objective. This keeps the library out of the business of reimplementing a
training loop that every user already has.

The returned callables take and return ``torch`` tensors and are differentiable,
so ``loss = task_loss + lambda * component(...)`` works directly. ``torch`` is
imported lazily inside each callable, so importing this module stays cheap.
"""

from __future__ import annotations

from typing import Any, Sequence

from fairlms.metrics.data import PromptPairs
from fairlms.mitigation.base import MitigationResult, Mitigator
from fairlms.mitigation.evidence import (
    AttributeLabeledVectors,
    GroupLabeledRecords,
    InfluenceScoredCorpus,
)

__all__ = [
    "AdversarialDebiasing",
    "CounterfactualInvarianceLoss",
    "GroupRegularizedObjective",
    "InfluenceGuidedSuppression",
]


class AdversarialDebiasing(Mitigator):
    """Gradient-reversal adversary predicting the protected attribute.

    Elazar and Goldberg (2018). The adversary is trained to read the protected
    attribute out of pooled hidden states; a gradient-reversal layer means the
    encoder is simultaneously trained to make that impossible.

    Returns ``(pooled, attribute_targets) -> loss``, already gradient-reversed,
    so adding it to the task loss trains the encoder adversarially. The adversary
    module is reachable as ``result.adversary`` for a separate optimizer, which
    is the usual way this is trained.

    Parameters
    ----------
    lambda_:
        Gradient-reversal strength.
    hidden_size:
        Width of the adversary's hidden layer.
    seed:
        Seed for adversary initialization.

    Examples
    --------
    >>> import torch
    >>> from fairlms.mitigation import AdversarialDebiasing, AttributeLabeledVectors
    >>> evidence = AttributeLabeledVectors(
    ...     axis="gender",
    ...     vectors=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
    ...     labels=["f", "f", "m", "m"], source="doctest",
    ... )
    >>> loss_term = AdversarialDebiasing().apply(None, evidence).result
    >>> pooled = torch.tensor(evidence.vectors, dtype=torch.float32)
    >>> float(loss_term(pooled, ["f", "f", "m", "m"])) > 0
    True
    """

    name = "adversarial_debiasing"
    category = "in"
    access = "white_box"
    architectures = ("encoder_only", "encoder_decoder")
    # Empty by design: this mitigator builds a loss term and reads nothing
    # from a model itself. The white-box demand of in-processing - the
    # training loop needs parameters and gradients - is carried by `access`,
    # which is checked whenever a model is supplied. Declaring capabilities
    # here would instead force a model to be passed just to build a loss.
    requires = frozenset()
    accepts = (AttributeLabeledVectors, GroupLabeledRecords)

    def __init__(self, lambda_: float = 1.0, hidden_size: int = 128, seed: int = 0):
        self.lambda_ = lambda_
        self.hidden_size = hidden_size
        self.seed = seed

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        import torch

        classes = sorted(set(evidence.labels))
        if len(classes) < 2:
            raise ValueError(
                f"{self.name}: the adversary needs at least two attribute values; "
                f"got {classes!r}."
            )
        n_features = (
            evidence.n_features
            if isinstance(evidence, AttributeLabeledVectors)
            else None
        )
        if n_features is None:
            raise ValueError(
                f"{self.name}: the adversary's input width must be known. Supply "
                f"AttributeLabeledVectors so n_features is declared."
            )

        torch.manual_seed(self.seed)
        adversary = torch.nn.Sequential(
            torch.nn.Linear(n_features, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, len(classes)),
        )
        index = {label: i for i, label in enumerate(classes)}
        lambda_ = self.lambda_

        class _Reverse(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.view_as(x)

            @staticmethod
            def backward(ctx, grad):
                return -lambda_ * grad

        def component(pooled: Any, attributes: Sequence[str]) -> Any:
            """Adversarial attribute-prediction loss, gradient-reversed."""
            unknown = sorted(set(attributes) - set(index))
            if unknown:
                raise ValueError(
                    f"{AdversarialDebiasing.name}: unseen attribute value(s) "
                    f"{unknown!r}; the adversary was built for {classes!r}."
                )
            if len(attributes) != pooled.shape[0]:
                raise ValueError(
                    f"{AdversarialDebiasing.name}: got {len(attributes)} attribute "
                    f"labels for {pooled.shape[0]} pooled rows."
                )
            targets = torch.tensor([index[a] for a in attributes], device=pooled.device)
            logits = adversary(_Reverse.apply(pooled))
            return torch.nn.functional.cross_entropy(logits, targets)

        component.adversary = adversary
        component.classes = tuple(classes)
        return self._result(
            component,
            axis=evidence.axis,
            attribute_values=list(classes),
            n_features=n_features,
            lambda_=self.lambda_,
            returns="loss component; the caller owns the training loop",
        )


class CounterfactualInvarianceLoss(Mitigator):
    """Symmetric KL between predictions on ``x`` and ``tau(x)``.

    Penalises any change in the predictive distribution under a counterfactual
    rewrite. Symmetric rather than one-directional so that neither branch is
    privileged as the reference.

    Returns ``(logits_factual, logits_counterfactual) -> loss``.

    Parameters
    ----------
    reduction:
        ``"mean"`` or ``"sum"`` over the batch.

    Examples
    --------
    >>> import torch
    >>> from fairlms.metrics import PromptPairs
    >>> from fairlms.mitigation import CounterfactualInvarianceLoss
    >>> pairs = PromptPairs(["he is a nurse"], ["she is a nurse"])
    >>> loss_term = CounterfactualInvarianceLoss().apply(None, pairs).result
    >>> identical = torch.tensor([[1.0, 2.0]])
    >>> round(float(loss_term(identical, identical)), 6)   # invariant: no penalty
    0.0
    >>> float(loss_term(identical, torch.tensor([[2.0, 1.0]]))) > 0
    True
    """

    name = "counterfactual_invariance_loss"
    category = "in"
    access = "white_box"
    architectures = ("encoder_only",)
    # Empty by design: this mitigator builds a loss term and reads nothing
    # from a model itself. The white-box demand of in-processing - the
    # training loop needs parameters and gradients - is carried by `access`,
    # which is checked whenever a model is supplied. Declaring capabilities
    # here would instead force a model to be passed just to build a loss.
    requires = frozenset()
    accepts = (PromptPairs,)

    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        import torch

        if self.reduction not in ("mean", "sum"):
            raise ValueError(
                f"reduction must be 'mean' or 'sum'; got {self.reduction!r}."
            )
        reduction = self.reduction

        def component(logits_factual: Any, logits_counterfactual: Any) -> Any:
            """Symmetric KL between the two branches' predictive distributions."""
            if logits_factual.shape != logits_counterfactual.shape:
                raise ValueError(
                    f"{CounterfactualInvarianceLoss.name}: the two branches must "
                    f"have the same shape; got {tuple(logits_factual.shape)} and "
                    f"{tuple(logits_counterfactual.shape)}."
                )
            log_p = torch.log_softmax(logits_factual, dim=-1)
            log_q = torch.log_softmax(logits_counterfactual, dim=-1)
            p, q = log_p.exp(), log_q.exp()
            per_row = ((p - q) * (log_p - log_q)).sum(dim=-1)
            return per_row.mean() if reduction == "mean" else per_row.sum()

        return self._result(
            component,
            n_pairs=len(evidence),
            reduction=reduction,
            returns="loss component; the caller owns the training loop",
        )


class GroupRegularizedObjective(Mitigator):
    """Differentiable TPR/FPR-gap surrogate added to the task loss.

    The hard rate gap is piecewise constant and has no useful gradient, so the
    surrogate replaces the indicator with the predicted positive probability.
    That makes the gap differentiable while keeping its meaning: the mean
    predicted score on the positive class, compared across groups.

    Says *objective* rather than the book's "group-regularized fine-tuning"
    because this ships a loss term, not a trainer.

    Returns ``(probabilities, labels, groups) -> loss``.

    Parameters
    ----------
    criterion:
        ``"equal_opportunity"`` penalises the TPR gap; ``"equalized_odds"``
        penalises TPR and FPR gaps jointly.

    Examples
    --------
    >>> import torch
    >>> from fairlms.mitigation import GroupLabeledRecords, GroupRegularizedObjective
    >>> records = GroupLabeledRecords(
    ...     axis="gender", groups=["f", "f", "m", "m"],
    ...     labels=["yes", "no", "yes", "no"],
    ...     label_name="outcome", source="doctest",
    ... )
    >>> loss_term = GroupRegularizedObjective().apply(None, records).result
    >>> probabilities = torch.tensor([0.8, 0.2, 0.8, 0.2])
    >>> groups, positives = ["f", "f", "m", "m"], [True, False, True, False]
    >>> round(float(loss_term(probabilities, positives, groups)), 6)  # no gap
    0.0
    """

    name = "group_regularized_objective"
    category = "in"
    access = "white_box"
    architectures = ("encoder_only",)
    # Empty by design: this mitigator builds a loss term and reads nothing
    # from a model itself. The white-box demand of in-processing - the
    # training loop needs parameters and gradients - is carried by `access`,
    # which is checked whenever a model is supplied. Declaring capabilities
    # here would instead force a model to be passed just to build a loss.
    requires = frozenset()
    accepts = (GroupLabeledRecords,)

    def __init__(self, criterion: str = "equal_opportunity"):
        self.criterion = criterion

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        import torch

        if self.criterion not in ("equal_opportunity", "equalized_odds"):
            raise ValueError(
                "criterion must be 'equal_opportunity' or 'equalized_odds'; got "
                f"{self.criterion!r}."
            )
        criterion = self.criterion

        def component(
            probabilities: Any, positives: Sequence[bool], groups: Sequence[str]
        ) -> Any:
            """Soft TPR (and optionally FPR) gap across groups."""
            if not (len(positives) == len(groups) == probabilities.shape[0]):
                raise ValueError(
                    f"{GroupRegularizedObjective.name}: probabilities, positives "
                    f"and groups must align; got {probabilities.shape[0]}, "
                    f"{len(positives)} and {len(groups)}."
                )
            gap = probabilities.new_zeros(())
            for wanted in (
                (True,) if criterion == "equal_opportunity" else (True, False)
            ):
                rates = []
                for group in sorted(set(groups)):
                    rows = [
                        i
                        for i, (g, p) in enumerate(zip(groups, positives))
                        if g == group and bool(p) is wanted
                    ]
                    if not rows:
                        raise ValueError(
                            f"{GroupRegularizedObjective.name}: group {group!r} has "
                            f"no {'positive' if wanted else 'negative'} rows in this "
                            f"batch, so its rate is undefined. Use a group-stratified "
                            f"sampler rather than letting the term silently skip it."
                        )
                    rates.append(probabilities[rows].mean())
                stacked = torch.stack(rates)
                gap = gap + (stacked.max() - stacked.min())
            return gap

        return self._result(
            component,
            axis=evidence.axis,
            criterion=criterion,
            groups=sorted(set(evidence.groups)),
            surrogate="mean predicted probability replaces the rate indicator",
            returns="loss component; the caller owns the training loop",
        )


class InfluenceGuidedSuppression(Mitigator):
    r"""Penalty ``lambda * sum |I(z', z)|`` over flagged harmful instances.

    A deliberate reduction of IF-Guide (Coalson et al., 2025): this ships the
    **objective only** and consumes **precomputed** influence scores. Estimating
    influence - Hessian inverse, EK-FAC, anything else - is an optional backend
    and never a core dependency, so the component is testable on scores the
    caller supplies and the influence machinery is not reimplemented here.

    Returns ``(per_example_loss) -> penalty``, which up-weights the training loss
    of flagged examples in proportion to their absolute influence, suppressing
    the model's tendency to fit them.

    Parameters
    ----------
    lambda_:
        Penalty strength.
    normalize:
        Divide influence scores by their maximum absolute value, so ``lambda_``
        means the same thing across corpora with different influence scales.

    Examples
    --------
    >>> import torch
    >>> from fairlms.mitigation import (
    ...     InfluenceGuidedSuppression, InfluenceScoredCorpus)
    >>> corpus = InfluenceScoredCorpus(
    ...     n_examples=4, flagged=[1, 3], influence=[2.0, 1.0],
    ...     source="doctest-precomputed",
    ... )
    >>> penalty = InfluenceGuidedSuppression().apply(None, corpus).result
    >>> losses = torch.tensor([1.0, 1.0, 1.0, 1.0])
    >>> float(penalty(losses, [0, 1, 2, 3]))   # 1.0 for row 1, 0.5 for row 3
    1.5
    """

    name = "influence_guided_suppression"
    category = "in"
    access = "white_box"
    architectures = ("decoder_only",)
    # Empty by design: this mitigator builds a loss term and reads nothing
    # from a model itself. The white-box demand of in-processing - the
    # training loop needs parameters and gradients - is carried by `access`,
    # which is checked whenever a model is supplied. Declaring capabilities
    # here would instead force a model to be passed just to build a loss.
    requires = frozenset()
    accepts = (InfluenceScoredCorpus,)

    def __init__(self, lambda_: float = 1.0, normalize: bool = True):
        self.lambda_ = lambda_
        self.normalize = normalize

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        import torch

        weights = [abs(value) for value in evidence.influence]
        largest = max(weights) if weights else 0.0
        if self.normalize and largest > 0:
            weights = [w / largest for w in weights]
        lookup = dict(zip(evidence.flagged, weights))
        lambda_ = self.lambda_
        n_examples = evidence.n_examples

        def component(per_example_loss: Any, indices: Sequence[int]) -> Any:
            """Influence-weighted penalty over the flagged rows of a batch."""
            if len(indices) != per_example_loss.shape[0]:
                raise ValueError(
                    f"{InfluenceGuidedSuppression.name}: got {len(indices)} indices "
                    f"for {per_example_loss.shape[0]} per-example losses."
                )
            out_of_range = [i for i in indices if not 0 <= i < n_examples]
            if out_of_range:
                raise ValueError(
                    f"{InfluenceGuidedSuppression.name}: indices "
                    f"{out_of_range[:5]!r} are outside the corpus range "
                    f"[0, {n_examples})."
                )
            scale = torch.tensor(
                [lookup.get(int(i), 0.0) for i in indices],
                dtype=per_example_loss.dtype,
                device=per_example_loss.device,
            )
            return lambda_ * (scale * per_example_loss).sum()

        return self._result(
            component,
            n_examples=evidence.n_examples,
            n_flagged=len(evidence.flagged),
            lambda_=lambda_,
            normalized=bool(self.normalize),
            influence_source=evidence.source,
            scope=(
                "objective only; influence scores are precomputed by the caller, "
                "the estimation half of IF-Guide is out of scope"
            ),
            returns="loss component; the caller owns the training loop",
        )
