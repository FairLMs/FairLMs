"""Registry helpers for dataset diagnostics."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Type

from .base import DatasetDiagnostic
from .representativeness import RepresentativenessBias
from .scoring import (
    ScorerCounterfactualSensitivity,
    ScorerMeanGap,
    ScorerRateGap,
    ScorerWasserstein1Gap,
)

DIAGNOSTIC_REGISTRY: Mapping[str, Type[DatasetDiagnostic]] = MappingProxyType(
    {
        RepresentativenessBias.name: RepresentativenessBias,
        ScorerCounterfactualSensitivity.name: ScorerCounterfactualSensitivity,
        ScorerMeanGap.name: ScorerMeanGap,
        ScorerRateGap.name: ScorerRateGap,
        ScorerWasserstein1Gap.name: ScorerWasserstein1Gap,
    }
)


def list_diagnostics() -> list[str]:
    """Return the registered diagnostic names in deterministic order."""

    return sorted(DIAGNOSTIC_REGISTRY)


def get_diagnostic(name: str) -> DatasetDiagnostic:
    """Instantiate the diagnostic registered under ``name``."""

    try:
        diagnostic_type = DIAGNOSTIC_REGISTRY[name]
    except KeyError as exc:
        alternatives = ", ".join(list_diagnostics())
        raise KeyError(
            f"Unknown diagnostic {name!r}. Available: {alternatives}"
        ) from exc
    return diagnostic_type()


__all__ = ["DIAGNOSTIC_REGISTRY", "get_diagnostic", "list_diagnostics"]
