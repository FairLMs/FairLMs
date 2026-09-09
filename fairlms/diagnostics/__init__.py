"""Dataset-first fairness diagnostics and explicit result-table audits."""

from .base import (
    DIAGNOSTIC_SCHEMA_VERSION,
    ComponentPlan,
    ComponentResult,
    DatasetDiagnostic,
    DiagnosticReport,
    DiagnosticStatus,
    ReportStatus,
)
from .evidence import (
    LabeledScoredGroups,
    PairedScores,
    RepresentationEvidence,
    ScoredGroups,
)
from .registry import DIAGNOSTIC_REGISTRY, get_diagnostic, list_diagnostics
from .representativeness import (
    RepresentativenessBias,
    audit_representativeness,
)
from .scoring import (
    ScoreRateDirection,
    ScoreRateTransform,
    ScorerCounterfactualSensitivity,
    ScorerMeanGap,
    ScorerRateGap,
    ScorerWasserstein1Gap,
    audit_scores,
)
from .spec import (
    DatasetAuditSpec,
    DesignStance,
    ReferenceDistribution,
    ReferencePurpose,
    TargetKind,
)

__all__ = [
    "DIAGNOSTIC_REGISTRY",
    "DIAGNOSTIC_SCHEMA_VERSION",
    "ComponentPlan",
    "ComponentResult",
    "DatasetAuditSpec",
    "DatasetDiagnostic",
    "DesignStance",
    "DiagnosticReport",
    "DiagnosticStatus",
    "LabeledScoredGroups",
    "PairedScores",
    "ReferenceDistribution",
    "ReferencePurpose",
    "ReportStatus",
    "RepresentationEvidence",
    "RepresentativenessBias",
    "ScoreRateDirection",
    "ScoreRateTransform",
    "ScoredGroups",
    "ScorerCounterfactualSensitivity",
    "ScorerMeanGap",
    "ScorerRateGap",
    "ScorerWasserstein1Gap",
    "TargetKind",
    "audit_representativeness",
    "audit_scores",
    "get_diagnostic",
    "list_diagnostics",
]
