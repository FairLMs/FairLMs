"""Registry helpers for dataset diagnostics."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Type

from .base import DatasetDiagnostic
from .construction import (
    FramingDisparity,
    LengthDisparity,
    MinimalPairResidual,
    OptionLengthBias,
    TemplateImbalance,
)
from .leakage import StereotypeLeakage
from .representativeness import RepresentativenessBias
from .scoring import (
    ScorerCounterfactualSensitivity,
    ScorerMeanGap,
    ScorerRateGap,
    ScorerWasserstein1Gap,
)

# Only implemented diagnostics are registered (D032). The three
# backend-dependent construction slots -- ``b_equiv``, ``b_gram`` and
# ``b_diff_dep`` -- have no class in this release: ``audit_construction``
# and ``audit_dataset`` synthesize a non-ready result for each of them from
# ``CONSTRUCTION_BACKEND_REQUIREMENTS``, so registering a stub for them
# would advertise a capability that does not exist. The backend is checked
# last, after the shared applicability precedence, so such a slot is
# ``blocked`` with its backend reason code only when it was requested and
# its required evidence view is present; otherwise it is ``not_applicable``
# for the ordinary reason. ``BACKEND_CONSTRUCTION_SLOTS``, not the reported
# status, is the way to enumerate the backend-dependent slots.
#
# Every registered diagnostic is constructible with zero arguments, which
# ``get_diagnostic`` requires. Several of them then report ``blocked`` for
# their own missing configuration -- ``b_min`` without an identity mask,
# ``b_opt`` without an option-role contrast, ``b_frame`` without a frame
# predicate, ``b_leak`` on text evidence without an extraction configuration.
# That is the designed outcome, not a defect: the configuration is an
# estimand declaration and is never inferred.
DIAGNOSTIC_REGISTRY: Mapping[str, Type[DatasetDiagnostic]] = MappingProxyType(
    {
        FramingDisparity.name: FramingDisparity,
        LengthDisparity.name: LengthDisparity,
        MinimalPairResidual.name: MinimalPairResidual,
        OptionLengthBias.name: OptionLengthBias,
        RepresentativenessBias.name: RepresentativenessBias,
        ScorerCounterfactualSensitivity.name: ScorerCounterfactualSensitivity,
        ScorerMeanGap.name: ScorerMeanGap,
        ScorerRateGap.name: ScorerRateGap,
        ScorerWasserstein1Gap.name: ScorerWasserstein1Gap,
        StereotypeLeakage.name: StereotypeLeakage,
        TemplateImbalance.name: TemplateImbalance,
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
