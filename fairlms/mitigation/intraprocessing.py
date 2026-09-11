"""Intra-processing mitigators: gray-box edits to a loaded model.

Every mitigator here returns a :class:`~fairlms.models.base.ModelAdapter`, which
is the load-bearing claim of the manuscript's introduction: because the result
*is* an adapter, any registered metric re-evaluates the mitigated model with no
modification at all.

.. warning::
   **Report removal relative to the probe family used.** A projection removes
   the attribute a *linear* probe could read. Gonen and Goldberg showed that
   this can hide bias rather than remove it: the geometry survives in clusters a
   linear probe no longer detects. Nothing here should be described as removing
   bias in absolute terms, and the provenance of every result records the probe
   family so the claim stays attached to its evidence.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from fairlms.metrics.data import GroupWordPairs, PromptPairs
from fairlms.mitigation.base import MitigationResult, Mitigator
from fairlms.mitigation.evidence import AttributeLabeledVectors, PromptSpec
from fairlms.models.base import LoadedModel, ModelAdapter

__all__ = [
    "IterativeNullspaceProjection",
    "ProjectedModelAdapter",
    "SelfDebiasedModelAdapter",
    "SelfDebiasing",
    "SubspaceProjection",
]


class ProjectedModelAdapter(ModelAdapter):
    """A :class:`ModelAdapter` that applies a projection to hidden states.

    Wraps the original adapter and installs a forward hook on the model's output
    embedding module, so the projection applies wherever the wrapped model is
    used. The original adapter is untouched and reachable as :attr:`base`.

    Because this satisfies the adapter interface, every metric in
    ``METRIC_REGISTRY`` accepts it unchanged.
    """

    def __init__(
        self,
        base: ModelAdapter,
        projection: np.ndarray,
        *,
        method: str,
        axis: str,
        probe_family: str,
    ):
        self.base = base
        self.projection = np.asarray(projection, dtype=float)
        self.method = method
        self.axis = axis
        self.probe_family = probe_family
        self.name = f"{getattr(base, 'name', type(base).__name__)}+{method}"
        # Mirror the tag the applicability matcher reads, so the mitigated model
        # profiles exactly as the model it wraps.
        self.task = getattr(base, "task", None)
        self._loaded: Optional[LoadedModel] = None

    def load(self) -> LoadedModel:
        if self._loaded is not None:
            return self._loaded

        import torch

        loaded = self.base.load()
        matrix = torch.tensor(
            self.projection, dtype=torch.float32, device=loaded.device
        )

        def project(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if not hasattr(tensor, "shape") or tensor.shape[-1] != matrix.shape[0]:
                return output
            projected = tensor @ matrix.to(tensor.dtype)
            if isinstance(output, tuple):
                return (projected,) + tuple(output[1:])
            return projected

        module = _embedding_module(loaded.model)
        self._handle = module.register_forward_hook(project)
        self._loaded = LoadedModel(
            name=self.name,
            tokenizer=loaded.tokenizer,
            model=loaded.model,
            device=loaded.device,
            task=loaded.task,
        )
        return self._loaded

    def remove(self) -> None:
        """Detach the projection hook, restoring the underlying model."""
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.remove()
            self._handle = None
        self._loaded = None

    def __repr__(self) -> str:
        return (
            f"ProjectedModelAdapter(base={self.base!r}, method={self.method!r}, "
            f"axis={self.axis!r})"
        )


def _embedding_module(model: Any):
    """Return the module whose output carries token representations."""
    for attribute in ("get_input_embeddings",):
        getter = getattr(model, attribute, None)
        if callable(getter):
            module = getter()
            if module is not None:
                return module
    raise TypeError(
        "cannot locate an input-embedding module on "
        f"{type(model).__name__}; a projection has nowhere to attach."
    )


def _nullspace_projection(basis: np.ndarray) -> np.ndarray:
    r"""Return ``P_perp = I - B (B^T B)^-1 B^T`` for column basis ``B``.

    Uses the pseudo-inverse, so a rank-deficient or near-collinear basis yields
    the projection onto the span actually spanned instead of raising on a
    singular Gram matrix.
    """
    basis = np.asarray(basis, dtype=float)
    if basis.ndim != 2:
        raise ValueError(f"basis must be 2-D (n_features, k); got {basis.shape}.")
    identity = np.eye(basis.shape[0])
    return identity - basis @ np.linalg.pinv(basis.T @ basis) @ basis.T


class SubspaceProjection(Mitigator):
    """Estimate a bias basis from paired evidence, then project it out.

    The basis is the top ``n_components`` principal directions of the
    within-pair difference vectors, and the returned adapter applies
    ``P_perp = I - B (B^T B)^-1 B^T`` to hidden states.

    Keeps its own name rather than the book's "projection-based debiasing",
    which also covers INLP - a separate component here.

    Parameters
    ----------
    n_components:
        Size of the estimated bias subspace.
    encode:
        Callable ``(model, texts) -> array`` producing representations. Required
        when the evidence is textual pairs; unused when vectors are supplied.

    Examples
    --------
    >>> import numpy as np
    >>> from fairlms.applicability import AccessLevel, ModelProfile
    >>> from fairlms.mitigation import AttributeLabeledVectors, SubspaceProjection
    >>> from fairlms.models.base import ModelAdapter
    >>> class Stub(ModelAdapter):
    ...     name = "stub"
    ...     task = "encoder"
    ...     def load(self): raise NotImplementedError
    >>> evidence = AttributeLabeledVectors(
    ...     axis="gender",
    ...     vectors=[[1.0, 0.0], [2.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
    ...     labels=["f", "f", "m", "m"], source="doctest",
    ... )
    >>> outcome = SubspaceProjection().apply(Stub(), evidence)
    >>> outcome.category
    'intra'
    >>> from fairlms.models.base import ModelAdapter
    >>> isinstance(outcome.result, ModelAdapter)      # re-usable by any metric
    True
    >>> np.round(outcome.result.projection, 6)        # first axis removed
    array([[0., 0.],
           [0., 1.]])
    """

    name = "subspace_projection"
    category = "intra"
    access = "gray_box"
    architectures = ("encoder_only", "encoder_decoder")
    requires = frozenset({"hidden_states"})
    accepts = (PromptPairs, GroupWordPairs, AttributeLabeledVectors)

    def __init__(self, n_components: int = 1, encode: Any = None):
        self.n_components = n_components
        self.encode = encode

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        if self.n_components < 1:
            raise ValueError(
                f"n_components must be at least 1; got {self.n_components!r}."
            )
        left, right = _pair_vectors(self, model, evidence)
        differences = left - right
        if differences.shape[0] < self.n_components:
            raise ValueError(
                f"{self.name}: {differences.shape[0]} pair(s) cannot support an "
                f"{self.n_components}-dimensional bias subspace."
            )

        centered = differences - differences.mean(axis=0, keepdims=True)
        _, singular, right_vectors = np.linalg.svd(centered, full_matrices=False)
        basis = right_vectors[: self.n_components].T
        projection = _nullspace_projection(basis)

        total = float((singular**2).sum())
        explained = (
            [float(s**2 / total) for s in singular[: self.n_components]]
            if total > 0
            else [0.0] * self.n_components
        )
        adapter = ProjectedModelAdapter(
            model,
            projection,
            method=self.name,
            axis=getattr(evidence, "axis", "declared-pairs"),
            probe_family="linear (PCA of paired differences)",
        )
        return self._result(
            adapter,
            n_components=self.n_components,
            n_pairs=int(differences.shape[0]),
            n_features=int(differences.shape[1]),
            explained_variance_ratio=explained,
            probe_family="linear (PCA of paired differences)",
            removal_claim=(
                "Removes the subspace a linear probe of this family reads. Not "
                "evidence of removal in absolute terms: projection can hide "
                "rather than remove bias (Gonen and Goldberg, 2019)."
            ),
        )


class IterativeNullspaceProjection(Mitigator):
    """Iteratively fit a linear attribute probe and project onto its nullspace.

    INLP (Ravfogel et al., 2020). Each round fits a linear classifier for the
    protected attribute, projects the representation onto that classifier's
    nullspace, and repeats until the probe is at chance or the iteration budget
    is spent. The composed projection is applied by the returned adapter.

    Stops early when probe accuracy reaches chance, and **reports the accuracy
    it actually reached**: claiming removal without the probe curve would be
    exactly the overstatement the literature warns about.

    Parameters
    ----------
    n_iterations:
        Maximum probe/project rounds.
    tolerance:
        Stop once probe accuracy is within this of the majority-class rate.
    seed:
        Seed for the probe's optimizer.

    Examples
    --------
    >>> from fairlms.mitigation import (
    ...     AttributeLabeledVectors, IterativeNullspaceProjection)
    >>> from fairlms.models.base import ModelAdapter
    >>> class Stub(ModelAdapter):
    ...     name = "stub"
    ...     task = "encoder"
    ...     def load(self): raise NotImplementedError
    >>> evidence = AttributeLabeledVectors(
    ...     axis="gender",
    ...     vectors=[[2.0, 0.1], [1.8, -0.1], [-2.0, 0.1], [-1.9, -0.2]],
    ...     labels=["f", "f", "m", "m"], source="doctest",
    ... )
    >>> outcome = IterativeNullspaceProjection(n_iterations=4).apply(
    ...     Stub(), evidence)
    >>> isinstance(outcome.result, ModelAdapter)
    True
    >>> outcome.provenance["probe_family"]
    'linear (logistic regression)'
    >>> 0.0 <= outcome.provenance["final_probe_accuracy"] <= 1.0
    True
    """

    name = "iterative_nullspace_projection"
    category = "intra"
    access = "gray_box"
    architectures = ("encoder_only", "encoder_decoder")
    requires = frozenset({"hidden_states"})
    accepts = (AttributeLabeledVectors,)

    def __init__(self, n_iterations: int = 8, tolerance: float = 0.02, seed: int = 0):
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.seed = seed

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        if self.n_iterations < 1:
            raise ValueError(
                f"n_iterations must be at least 1; got {self.n_iterations!r}."
            )
        vectors = np.array(evidence.vectors, dtype=float)
        classes = sorted(set(evidence.labels))
        if len(classes) != 2:
            raise ValueError(
                f"{self.name}: INLP fits a binary attribute probe; got "
                f"{len(classes)} attribute values {classes!r}."
            )
        targets = np.array(
            [1.0 if label == classes[1] else 0.0 for label in evidence.labels]
        )
        chance = float(max(targets.mean(), 1.0 - targets.mean()))

        n_features = vectors.shape[1]
        composed = np.eye(n_features)
        current = vectors
        history = []
        for _ in range(self.n_iterations):
            weights = _fit_linear_probe(current, targets, seed=self.seed)
            accuracy = _probe_accuracy(current, targets, weights)
            history.append(accuracy)
            if accuracy <= chance + self.tolerance:
                break
            direction = weights.reshape(-1, 1)
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                break
            step = _nullspace_projection(direction / norm)
            composed = step @ composed
            current = current @ step.T

        adapter = ProjectedModelAdapter(
            model,
            composed,
            method=self.name,
            axis=evidence.axis,
            probe_family="linear (logistic regression)",
        )
        return self._result(
            adapter,
            axis=evidence.axis,
            n_iterations_run=len(history),
            n_iterations_max=self.n_iterations,
            probe_accuracy_history=[float(a) for a in history],
            final_probe_accuracy=float(history[-1]) if history else None,
            majority_class_rate=chance,
            reached_chance=bool(history and history[-1] <= chance + self.tolerance),
            probe_family="linear (logistic regression)",
            removal_claim=(
                "Attribute is no longer linearly decodable by this probe family, "
                "to the reported accuracy. Not evidence of removal in absolute "
                "terms: projection can hide rather than remove bias (Gonen and "
                "Goldberg, 2019)."
            ),
        )


def _fit_linear_probe(
    features: np.ndarray, targets: np.ndarray, *, seed: int, steps: int = 200
) -> np.ndarray:
    """Fit a logistic probe by gradient descent.

    Written out rather than delegated to scikit-learn so that INLP runs in the
    base install; the probe is a means to a nullspace, not a model to ship.
    """
    rng = np.random.default_rng(seed)
    weights = rng.normal(scale=0.01, size=features.shape[1])
    n = max(len(targets), 1)
    for _ in range(steps):
        logits = features @ weights
        predictions = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        gradient = features.T @ (predictions - targets) / n
        weights -= 0.5 * gradient
    return weights


def _probe_accuracy(
    features: np.ndarray, targets: np.ndarray, weights: np.ndarray
) -> float:
    predictions = (features @ weights) > 0
    return float((predictions == (targets > 0.5)).mean())


class SelfDebiasedModelAdapter(ModelAdapter):
    """A :class:`ModelAdapter` that self-diagnoses, then rescales next-token scores.

    Wraps the original adapter. :meth:`load` returns the underlying
    :class:`LoadedModel` so every metric works unchanged; the debiasing
    behaviour is exposed through :meth:`generate`, which is what a decoder-only
    generative metric drives.
    """

    def __init__(
        self,
        base: ModelAdapter,
        spec: PromptSpec,
        *,
        decay: float,
        max_new_tokens: int,
    ):
        self.base = base
        self.spec = spec
        self.decay = decay
        self.max_new_tokens = max_new_tokens
        self.name = f"{getattr(base, 'name', type(base).__name__)}+self_debiasing"
        self.task = getattr(base, "task", None)
        self._loaded: Optional[LoadedModel] = None

    def load(self) -> LoadedModel:
        if self._loaded is None:
            self._loaded = self.base.load()
        return self._loaded

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate with the diagnosed attribute's token scores scaled down.

        Runs the input twice: once under the self-diagnosis template and once
        plain. Tokens the diagnosis pass favours over the plain pass are damped
        by ``exp(-decay * delta)``, following Schick et al. (2021).
        """
        import torch

        loaded = self.load()
        tokenizer, model = loaded.tokenizer, loaded.model
        max_new_tokens = kwargs.pop("max_new_tokens", self.max_new_tokens)
        if kwargs:
            raise TypeError(
                f"generate() got unexpected keyword argument(s): "
                f"{', '.join(sorted(kwargs))}."
            )

        diagnosis_prompt = self.spec.render(prompt)[0]
        generated = prompt
        for _ in range(max_new_tokens):
            plain = _next_token_logits(model, tokenizer, generated, loaded.device)
            biased = _next_token_logits(
                model,
                tokenizer,
                diagnosis_prompt + generated[len(prompt) :],
                loaded.device,
            )
            delta = torch.clamp(biased - plain, min=0.0)
            adjusted = plain - self.decay * delta
            next_id = int(torch.argmax(adjusted).item())
            if next_id == getattr(tokenizer, "eos_token_id", None):
                break
            generated += tokenizer.decode([next_id])
        return generated

    def __repr__(self) -> str:
        return f"SelfDebiasedModelAdapter(base={self.base!r}, decay={self.decay!r})"


def _next_token_logits(model: Any, tokenizer: Any, text: str, device: Any):
    import torch

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        return model(**inputs).logits[0, -1].float()


class SelfDebiasing(Mitigator):
    """Self-diagnosis prompt, then rescale the next-token distribution.

    Schick et al. (2021). The model is asked to describe the attribute itself,
    and tokens the diagnosis pass favours are damped in the plain pass. Needs no
    weight update, which is what makes it intra-processing rather than
    in-processing.

    Parameters
    ----------
    decay:
        Strength of the damping applied to diagnosed tokens.
    max_new_tokens:
        Default generation length for the returned adapter.

    Examples
    --------
    >>> from fairlms.mitigation import PromptSpec, SelfDebiasing
    >>> from fairlms.models.base import ModelAdapter
    >>> class Stub(ModelAdapter):
    ...     name = "stub"
    ...     task = "causal"
    ...     def load(self): raise NotImplementedError
    >>> spec = PromptSpec(
    ...     templates=["The following text is biased. {query}"],
    ...     attribute="gender",
    ... )
    >>> outcome = SelfDebiasing().apply(Stub(), spec)
    >>> isinstance(outcome.result, ModelAdapter)
    True
    >>> outcome.result.task            # profiles as the model it wraps
    'causal'
    """

    name = "self_debiasing"
    category = "intra"
    access = "gray_box"
    architectures = ("decoder_only",)
    requires = frozenset({"token_logprobs", "free_generation"})
    accepts = (PromptSpec,)

    def __init__(self, decay: float = 50.0, max_new_tokens: int = 20):
        self.decay = decay
        self.max_new_tokens = max_new_tokens

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        if self.decay < 0:
            raise ValueError(f"decay must be non-negative; got {self.decay!r}.")
        adapter = SelfDebiasedModelAdapter(
            model,
            evidence,
            decay=self.decay,
            max_new_tokens=self.max_new_tokens,
        )
        return self._result(
            adapter,
            attribute=evidence.attribute,
            n_templates=len(evidence.templates),
            templates=list(evidence.templates),
            removal_claim=(
                "Damps tokens the model's own diagnosis pass favours. Acts on the "
                "output distribution only; the underlying representations are "
                "unchanged."
            ),
        )


def _pair_vectors(mitigator: Mitigator, model: Any, evidence: Any):
    """Return aligned ``(left, right)`` representation arrays from paired evidence."""
    if isinstance(evidence, AttributeLabeledVectors):
        classes = sorted(set(evidence.labels))
        if len(classes) != 2:
            raise ValueError(
                f"{mitigator.name}: paired estimation needs exactly two attribute "
                f"values; got {classes!r}."
            )
        vectors = np.asarray(evidence.vectors, dtype=float)
        left = vectors[[i for i, x in enumerate(evidence.labels) if x == classes[0]]]
        right = vectors[[i for i, x in enumerate(evidence.labels) if x == classes[1]]]
        if left.shape[0] != right.shape[0]:
            raise ValueError(
                f"{mitigator.name}: the two attribute values must have equally "
                f"many rows to form pairs; got {left.shape[0]} and "
                f"{right.shape[0]}. Supply AttributeLabeledVectors with matched "
                f"rows, or use PromptPairs."
            )
        return left, right

    if isinstance(evidence, PromptPairs):
        left_texts, right_texts = list(evidence.factual), list(evidence.counterfactual)
    else:  # GroupWordPairs
        left_texts, right_texts = list(evidence.group_1), list(evidence.group_2)

    if mitigator.encode is None:
        raise ValueError(
            f"{mitigator.name}: textual pair evidence needs an encoder. Pass "
            f"encode=(model, texts) -> array, or supply "
            f"AttributeLabeledVectors with representations already computed."
        )
    left = np.asarray(mitigator.encode(model, left_texts), dtype=float)
    right = np.asarray(mitigator.encode(model, right_texts), dtype=float)
    if left.shape != right.shape:
        raise ValueError(
            f"{mitigator.name}: encode returned mismatched shapes {left.shape} and "
            f"{right.shape} for the two sides of the pair set."
        )
    return left, right
