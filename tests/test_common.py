"""Conformance suite: contract checks every metric in the registry must satisfy.

This is the fairllms analogue of scikit-learn's ``check_estimator`` /
``parametrize_with_checks``. It is the mechanism that *keeps* the public API
uniform as metrics are added — a new metric that forgets the parameter protocol
fails here rather than surprising a user.

These checks are deliberately model-free so they run fast, offline, and in CI.
"""

import inspect

import pytest

from fairllms.metrics import (
    METRIC_REGISTRY,
    FairnessMetric,
    MetricResult,
    get_metric,
    list_metrics,
)

ALL_METRICS = list_metrics()

# Every metric is now on the strict ``compute(model, data)`` contract, so the
# strict checks below run over the whole registry. A newly added metric that
# forgets the shape fails here.
MIGRATED = ALL_METRICS


def test_registry_is_not_empty():
    assert ALL_METRICS, "metric registry is empty"


@pytest.mark.parametrize("name", ALL_METRICS)
def test_registry_name_matches_class_attribute(name):
    """``METRIC_REGISTRY`` keys must match each class's declared ``name``."""
    assert METRIC_REGISTRY[name].name == name


@pytest.mark.parametrize("name", ALL_METRICS)
def test_instantiable_with_no_arguments(name):
    """Every metric must construct with defaults, as sklearn estimators do."""
    metric = get_metric(name)
    assert isinstance(metric, FairnessMetric)


@pytest.mark.parametrize("name", ALL_METRICS)
def test_init_stores_params_verbatim(name):
    """``__init__`` must store each argument unmodified under its own name.

    sklearn requires this so that ``get_params`` can reconstruct an estimator.
    It also forbids validation or computation in ``__init__``.
    """
    cls = METRIC_REGISTRY[name]
    metric = cls()
    for param in cls._param_names():
        assert hasattr(metric, param), (
            f"{name}: __init__ parameter {param!r} is not stored as "
            f"self.{param}, so get_params() cannot see it"
        )


@pytest.mark.parametrize("name", ALL_METRICS)
def test_get_params_roundtrips(name):
    """``type(m)(**m.get_params())`` must reproduce an equivalent metric."""
    metric = get_metric(name)
    params = metric.get_params()
    clone = type(metric)(**params)
    assert clone.get_params() == params


@pytest.mark.parametrize("name", ALL_METRICS)
def test_set_params_rejects_unknown(name):
    metric = get_metric(name)
    with pytest.raises(ValueError, match="Invalid parameter"):
        metric.set_params(definitely_not_a_real_param=1)


@pytest.mark.parametrize("name", ALL_METRICS)
def test_set_params_updates_and_returns_self(name):
    metric = get_metric(name)
    params = metric.get_params()
    if not params:
        pytest.skip(f"{name} has no configuration parameters")
    key = sorted(params)[0]
    sentinel = object()
    assert metric.set_params(**{key: sentinel}) is metric
    assert metric.get_params()[key] is sentinel


@pytest.mark.parametrize("name", ALL_METRICS)
def test_repr_shows_non_default_params(name):
    """Default repr is bare; a changed param must appear in it."""
    cls = METRIC_REGISTRY[name]
    assert repr(cls()) == f"{cls.__name__}()"

    params = cls().get_params()
    if not params:
        pytest.skip(f"{name} has no configuration parameters")
    key = sorted(params)[0]
    metric = cls().set_params(**{key: "SENTINEL_VALUE"})
    assert key in repr(metric)
    assert "SENTINEL_VALUE" in repr(metric)


@pytest.mark.parametrize("name", ALL_METRICS)
def test_compute_exists_and_is_documented(name):
    cls = METRIC_REGISTRY[name]
    assert callable(cls.compute)
    assert cls.compute.__doc__ or cls.__doc__, f"{name} has no docstring"


@pytest.mark.parametrize("name", MIGRATED)
def test_migrated_compute_signature(name):
    """Migrated metrics expose exactly ``compute(model, data, *, ...)``.

    The uniform two-positional-argument shape is what makes generic tooling
    over metrics possible; it is the whole point of the refactor.
    """
    params = list(inspect.signature(METRIC_REGISTRY[name].compute).parameters.values())
    positional = [
        p.name
        for p in params
        if p.name != "self" and p.kind is p.POSITIONAL_OR_KEYWORD
    ]
    assert positional == ["model", "data"], (
        f"{name}.compute() positional args are {positional}, expected "
        f"['model', 'data']"
    )


@pytest.mark.parametrize("name", ALL_METRICS)
def test_unknown_kwarg_rejection_is_wired(name):
    """Every metric must reject unknown keywords rather than ignore them.

    Tested at the mechanism level so it applies uniformly: metrics validate their
    data before reaching the keyword check, and building valid data for all 33
    would require loading models.
    """
    metric = get_metric(name)
    with pytest.raises(TypeError, match="unexpected keyword"):
        metric._reject_unknown_kwargs({"definitely_not_a_kwarg": 1})


@pytest.mark.parametrize("name", ALL_METRICS)
def test_declared_params_are_accepted_kwargs(name):
    """A metric's own config names must survive its unknown-kwarg check.

    Guards against a metric declaring ``__init__(self, *, k=3)`` but forgetting
    to list ``"k"`` as accepted, which would make the deprecated per-call
    override path raise spuriously.
    """
    metric = get_metric(name)
    params = metric.get_params()
    if not params:
        pytest.skip(f"{name} has no configuration parameters")
    # Should not raise: every declared param is an accepted keyword.
    metric._reject_unknown_kwargs(params, *params)


# End-to-end keyword rejection, for the metrics whose valid data is cheap to
# build (no model download required).
def _cheap_cases():
    import numpy as np

    from fairllms.metrics import GroupPredictions, ScorePair, VectorSets

    rng = np.random.default_rng(0)
    return {
        "weat": VectorSets(*(rng.normal(size=(4, 8)) for _ in range(4))),
        "accuracy_disparity": ScorePair([1.0, 0.0], [0.0, 1.0]),
        "inference_bias_score": [(1, 1), (0, 0)],
        "fair_inference_score": [{"entailment": 0.1, "neutral": 0.8, "contradiction": 0.1}],
        "context_based_disparity": [
            {"cond": "disambig", "output": "unknown", "expected": "target"}
        ],
        "equal_opportunity_gap": GroupPredictions([1, 1, 0], [1, 0, 0], ["A", "B", "A"]),
    }


@pytest.mark.parametrize("name", sorted(_cheap_cases()))
def test_rejects_unknown_kwargs_end_to_end(name):
    metric = get_metric(name)
    with pytest.raises(TypeError, match="unexpected keyword"):
        metric.compute(None, _cheap_cases()[name], n_bootstrp=20)


@pytest.mark.parametrize("name", sorted(_cheap_cases()))
def test_model_free_metrics_return_float_score(name):
    """The model-free metrics must produce a real score with no model at all."""
    result = get_metric(name).compute(None, _cheap_cases()[name])
    assert isinstance(result, MetricResult)
    assert isinstance(float(result), float)


def test_metric_result_is_float_convertible():
    result = MetricResult(score=0.25, details={"p_value": 0.1})
    assert float(result) == 0.25
    assert "score=0.25" in repr(result)


def test_get_metric_unknown_name_lists_alternatives():
    with pytest.raises(KeyError) as excinfo:
        get_metric("not_a_metric")
    assert "weat" in str(excinfo.value)
