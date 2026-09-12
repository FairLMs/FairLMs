"""Intra-processing mitigators: gray-box edits to a loaded model.

Every mitigator here returns a :class:`~fairlms.models.base.ModelAdapter`, so supported metrics can evaluate the edited behavior through the same API.
Capabilities describe the edited outputs, not capabilities of the base model.

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

    Installs a forward hook at an explicitly named representation layer.
    The base and edited adapters share the loaded model: while the hook is
    installed, callers of the base see the projection too. Use ``remove()``
    before a baseline; ``compare_before_after`` manages the hook lifecycle.
    Do not use a shared base concurrently while the hook is active.
    """

    def __init__(
        self,
        base: ModelAdapter,
        projection: np.ndarray,
        *,
        method: str,
        axis: str,
        probe_family: str,
        layer: str,
    ):
        if not isinstance(layer, str) or not layer.strip():
            raise ValueError("A named representation layer is required.")
        self.layer = layer
        self.base = base
        self.projection = np.array(projection, dtype=float, copy=True)
        if (
            self.projection.ndim != 2
            or self.projection.shape[0] != self.projection.shape[1]
            or not np.isfinite(self.projection).all()
        ):
            raise ValueError("projection must be a finite square matrix.")
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
        matrix = torch.tensor(self.projection, dtype=torch.float32)

        def project_tensor(tensor):
            if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2:
                raise TypeError(
                    f"Layer {self.layer!r} does not expose token/row representations."
                )
            if tensor.shape[-1] != matrix.shape[0]:
                raise ValueError(
                    f"Projection width {matrix.shape[0]} does not match layer {self.layer!r} width {tensor.shape[-1]}."
                )
            return tensor @ matrix.to(device=tensor.device, dtype=tensor.dtype)

        def project(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                return project_tensor(output)
            if isinstance(output, tuple) and output:
                return (project_tensor(output[0]),) + tuple(output[1:])
            if isinstance(output, dict) and "last_hidden_state" in output:
                import copy

                result = copy.copy(output)
                result["last_hidden_state"] = project_tensor(
                    output["last_hidden_state"]
                )
                if output.get("hidden_states") is not None:
                    result["hidden_states"] = tuple(output["hidden_states"][:-1]) + (
                        result["last_hidden_state"],
                    )
                return result
            raise TypeError(
                f"Layer {self.layer!r} returned an unsupported representation type {type(output).__name__}."
            )

        module = _embedding_module(loaded.model, self.layer)
        weight = getattr(module, "weight", None)
        if (
            isinstance(weight, torch.Tensor)
            and weight.ndim == 2
            and self.layer == "input_embeddings"
            and weight.shape[1] != matrix.shape[0]
        ):
            raise ValueError(
                "Projection dimension does not match the declared input_embeddings layer."
            )
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


def _embedding_module(model: Any, layer: str):
    """Resolve the explicitly declared representation intervention site."""
    if layer == "input_embeddings":
        getter = getattr(model, "get_input_embeddings", None)
        module = getter() if callable(getter) else None
    else:
        try:
            module = model.get_submodule(layer)
        except (AttributeError, KeyError) as exc:
            raise ValueError(f"Model has no representation layer {layer!r}.") from exc
    if module is None:
        raise TypeError(f"Cannot locate representation layer {layer!r}.")
    return module


def _declared_layer(mitigator, evidence):
    supplied = getattr(evidence, "representation_layer", None)
    configured = mitigator.layer
    if supplied is not None and configured is not None and supplied != configured:
        raise ValueError(
            "Evidence representation_layer differs from the requested intervention layer."
        )
    layer = configured or supplied
    if not isinstance(layer, str) or not layer.strip():
        raise ValueError(
            "Declare the representation layer via layer= or evidence.representation_layer; it is never inferred from vector width."
        )
    return layer


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

    The basis is the top ``n_components`` uncentered singular directions of the
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
    ...     representation_layer="input_embeddings", pooling="token",
    ...     pair_ids=["p1", "p2", "p1", "p2"],
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

    def __init__(
        self, n_components: int = 1, encode: Any = None, layer: Optional[str] = None
    ):
        self.n_components = n_components
        self.encode = encode
        self.layer = layer

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        if self.n_components < 1:
            raise ValueError(
                f"n_components must be at least 1; got {self.n_components!r}."
            )
        layer = _declared_layer(self, evidence)
        left, right = _pair_vectors(self, model, evidence)
        differences = left - right
        if differences.shape[0] < self.n_components:
            raise ValueError(
                f"{self.name}: {differences.shape[0]} pair(s) cannot support an "
                f"{self.n_components}-dimensional bias subspace."
            )

        # Within-pair centering yields +difference/2 and -difference/2.
        # SVD of raw differences has that same span; centering the difference
        # rows again would incorrectly erase a shared protected direction.
        _, singular, right_vectors = np.linalg.svd(differences, full_matrices=False)
        rank = np.linalg.matrix_rank(differences)
        if rank < self.n_components:
            raise ValueError(
                f"Paired differences have rank {rank}; cannot estimate {self.n_components} components."
            )
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
            probe_family="linear (SVD of within-pair differences)",
            layer=layer,
        )
        return self._result(
            adapter,
            representation_layer=layer,
            pooling=getattr(evidence, "pooling", None),
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
    ...     representation_layer="input_embeddings", pooling="token",
    ...     pair_ids=["p1", "p2", "p1", "p2"],
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

    def __init__(
        self,
        n_iterations: int = 8,
        tolerance: float = 0.02,
        seed: int = 0,
        layer: Optional[str] = None,
        validation_fraction: float = 0.25,
    ):
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.seed = seed
        self.layer = layer
        self.validation_fraction = validation_fraction

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        if (
            isinstance(self.n_iterations, bool)
            or not isinstance(self.n_iterations, int)
            or self.n_iterations < 1
        ):
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
        from sklearn.model_selection import train_test_split

        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one.")
        if not np.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and non-negative.")
        if min(sum(targets == value) for value in (0.0, 1.0)) < 2:
            raise ValueError(
                "INLP needs at least two rows per group for held-out probe evaluation."
            )
        layer = _declared_layer(self, evidence)
        validation_size = max(2, int(np.ceil(len(targets) * self.validation_fraction)))
        if validation_size > len(targets) - 2:
            raise ValueError(
                "validation_fraction leaves insufficient probe training rows."
            )
        train, validation = train_test_split(
            np.arange(len(targets)),
            test_size=validation_size,
            stratify=targets,
            random_state=self.seed,
        )
        chance = float(
            max((targets[validation] == 0).mean(), (targets[validation] == 1).mean())
        )
        n_features = vectors.shape[1]
        composed = np.eye(n_features)
        directions, history = [], []
        for _ in range(self.n_iterations):
            current = vectors @ composed
            weights = _fit_linear_probe(current[train], targets[train], seed=self.seed)
            accuracy = _probe_accuracy(
                current[validation], targets[validation], weights
            )
            history.append(accuracy)
            if accuracy <= chance + self.tolerance:
                break
            direction = composed @ weights
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                break
            directions.append(direction / norm)
            # One orthogonal projection onto the intersection of nullspaces.
            composed = _nullspace_projection(np.stack(directions, axis=1))

        final_vectors = vectors @ composed
        final_weights = _fit_linear_probe(
            final_vectors[train], targets[train], seed=self.seed
        )
        final_accuracy = _probe_accuracy(
            final_vectors[validation], targets[validation], final_weights
        )
        adapter = ProjectedModelAdapter(
            model,
            composed,
            method=self.name,
            axis=evidence.axis,
            probe_family="linear (logistic regression)",
            layer=layer,
        )
        return self._result(
            adapter,
            axis=evidence.axis,
            representation_layer=layer,
            pooling=getattr(evidence, "pooling", None),
            n_iterations_run=len(directions),
            n_iterations_max=self.n_iterations,
            probe_accuracy_history=[float(a) for a in history],
            final_probe_accuracy=float(final_accuracy),
            probe_evaluation="held-out validation; used for stopping, not an independent test set",
            n_probe_train=len(train),
            n_probe_validation=len(validation),
            majority_class_rate=chance,
            reached_chance=bool(final_accuracy <= chance + self.tolerance),
            probe_family="linear (logistic regression)",
            removal_claim=(
                "Probe performance on the final projected representation is reported; "
                "this is not evidence of absolute bias removal (Gonen and Goldberg, 2019)."
            ),
        )


def _fit_linear_probe(
    features: np.ndarray, targets: np.ndarray, *, seed: int, steps: int = 200
) -> np.ndarray:
    """Fit a logistic probe by gradient descent.

    The no-intercept logistic probe uses a fixed optimizer schedule. Its
    held-out accuracy is reported as a diagnostic, not a fairness certificate.
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
    """An adapter exposing edited generation through its loaded model as well.

    Only generation is declared: unchanged internal activations/logits must not
    be reported as if they had been edited by this intervention.
    """

    def __init__(self, base, spec, *, decay, max_new_tokens, epsilon=0.01, seed=0):
        from fairlms.applicability import ModelProfile, AccessLevel

        self.base, self.spec = base, spec
        self.decay, self.max_new_tokens = decay, max_new_tokens
        self.epsilon, self.seed = epsilon, seed
        self.name = f"{getattr(base, 'name', type(base).__name__)}+self_debiasing"
        self.task = "causal"
        self.profile = ModelProfile(
            architecture="decoder_only",
            capabilities=frozenset({"free_generation", "local_tokenizer"}),
            access=AccessLevel.GRAY_BOX,
            task="causal",
        )
        self._loaded = None

    def load(self):
        if self._loaded is None:
            from ._generation import SelfDebiasedDecoder

            loaded = self.base.load()
            edited = SelfDebiasedDecoder(
                loaded.model,
                loaded.tokenizer,
                self.spec,
                decay=self.decay,
                epsilon=self.epsilon,
                max_new_tokens=self.max_new_tokens,
                seed=self.seed,
            )
            self._loaded = LoadedModel(
                name=self.name,
                tokenizer=loaded.tokenizer,
                model=edited,
                device=loaded.device,
                task=self.task,
                profile=self.profile,
            )
        return self._loaded

    def generate(self, prompt: str, **kwargs):
        loaded = self.load()
        inputs = loaded.tokenizer(prompt, return_tensors="pt").to(loaded.device)
        result = loaded.model.generate(**inputs, **kwargs)
        return loaded.tokenizer.decode(result[0], skip_special_tokens=True)

    def __repr__(self):
        return f"SelfDebiasedModelAdapter(base={self.base!r}, decay={self.decay!r})"


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

    def __init__(
        self,
        decay: float = 50.0,
        max_new_tokens: int = 20,
        epsilon: float = 0.01,
        seed: Optional[int] = 0,
    ):
        self.decay = decay
        self.max_new_tokens = max_new_tokens
        self.epsilon = epsilon
        self.seed = seed

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        import math

        if not math.isfinite(self.decay) or self.decay < 0:
            raise ValueError(
                f"decay must be non-negative and finite; got {self.decay!r}."
            )
        if not 0 < self.epsilon <= 1:
            raise ValueError("epsilon must be in (0, 1].")
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or self.max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be a positive integer.")
        adapter = SelfDebiasedModelAdapter(
            model,
            evidence,
            decay=self.decay,
            max_new_tokens=self.max_new_tokens,
            epsilon=self.epsilon,
            seed=self.seed,
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
        if evidence.pair_ids is None:
            raise ValueError(
                "SubspaceProjection needs explicit pair_ids for attribute-labeled vectors; group row order is not pairing evidence."
            )
        ids = list(dict.fromkeys(evidence.pair_ids))
        left, right = [], []
        for pair_id in ids:
            rows = [i for i, value in enumerate(evidence.pair_ids) if value == pair_id]
            if len(rows) != 2 or {evidence.labels[i] for i in rows} != set(classes):
                raise ValueError(
                    "Each pair_id must identify exactly two rows, one from each attribute value."
                )
            left.append(
                vectors[next(i for i in rows if evidence.labels[i] == classes[0])]
            )
            right.append(
                vectors[next(i for i in rows if evidence.labels[i] == classes[1])]
            )
        return np.asarray(left), np.asarray(right)

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
    if left.ndim != 2 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("encode must return finite 2-D representation arrays.")
    if left.shape != right.shape:
        raise ValueError(
            f"{mitigator.name}: encode returned mismatched shapes {left.shape} and "
            f"{right.shape} for the two sides of the pair set."
        )
    return left, right
