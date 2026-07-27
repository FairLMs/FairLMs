"""Public sklearn-style metric API.

Every metric exposes ``compute(model, dataset=None, **kwargs) -> MetricResult``.

Modules are grouped by metric family (e.g. ``similarity_based`` for WEAT/SEAT/CEAT).
"""

from fairLLMs.metrics.algorithmic_disparity import (
    LexicalFrequencyProportion,
    MorphologicalChoiceDivergence,
)
from fairLLMs.metrics.attention_head import (
    GradientBasedBiasEstimation,
    NaturalIndirectEffect,
)
from fairLLMs.metrics.base import FairnessMetric, MetricResult
from fairLLMs.metrics.counterfactual_fairness import (
    CounterfactualFairnessScore,
    CounterfactualRobustness,
)
from fairLLMs.metrics.demographic_representation import (
    DemographicNextTokenProportion,
    DemographicRepresentationDivergence,
)
from fairLLMs.metrics.encoder_decoder_extrinsic import (
    CounterfactualAucScore,
    InferenceBiasScore,
    NormalizedPositionDistance,
    TranslationSimilarityScore,
)
from fairLLMs.metrics.encoder_decoder_stereotypical import (
    StereotypicalDivergence,
    StereotypicalValueAttribution,
)
from fairLLMs.metrics.encoder_extrinsic import (
    ContextBasedDisparityScore,
    EqualOpportunityGap,
    FairInferenceScore,
)
from fairLLMs.metrics.masked_token import (
    ContrastBasedScore,
    DiscoveryOfCorrelationsScore,
    LogProbabilityBiasScore,
)
from fairLLMs.metrics.performance_disparity import (
    AccuracyDisparity,
    BiasAmplifierScore,
    SensitiveNameSimilarity,
)
from fairLLMs.metrics.pseudo_log_likelihood import (
    AllUnmaskedLikelihoodAttentionScore,
    AllUnmaskedLikelihoodScore,
    ContextAssociationTestScore,
    CrowSPairsScore,
    PseudoLogLikelihoodScore,
)
from fairLLMs.metrics.similarity_based import CEAT, SEAT, WEAT
from fairLLMs.metrics.stereotypical_association import (
    CooccurrenceAssociation,
    StereotypicalLogLikelihood,
)

# Short aliases used in papers / book chapters
CPS = CrowSPairsScore
LPBS = LogProbabilityBiasScore
CAT = ContextAssociationTestScore
DisCo = DiscoveryOfCorrelationsScore
CBS = ContrastBasedScore
PLL = PseudoLogLikelihoodScore
AUL = AllUnmaskedLikelihoodScore
AULA = AllUnmaskedLikelihoodAttentionScore

METRIC_REGISTRY = {
    CrowSPairsScore.name: CrowSPairsScore,
    LogProbabilityBiasScore.name: LogProbabilityBiasScore,
    WEAT.name: WEAT,
    SEAT.name: SEAT,
    CEAT.name: CEAT,
    PseudoLogLikelihoodScore.name: PseudoLogLikelihoodScore,
    AllUnmaskedLikelihoodScore.name: AllUnmaskedLikelihoodScore,
    AllUnmaskedLikelihoodAttentionScore.name: AllUnmaskedLikelihoodAttentionScore,
    ContextAssociationTestScore.name: ContextAssociationTestScore,
    DiscoveryOfCorrelationsScore.name: DiscoveryOfCorrelationsScore,
    ContrastBasedScore.name: ContrastBasedScore,
    FairInferenceScore.name: FairInferenceScore,
    EqualOpportunityGap.name: EqualOpportunityGap,
    ContextBasedDisparityScore.name: ContextBasedDisparityScore,
    CounterfactualAucScore.name: CounterfactualAucScore,
    InferenceBiasScore.name: InferenceBiasScore,
    TranslationSimilarityScore.name: TranslationSimilarityScore,
    NormalizedPositionDistance.name: NormalizedPositionDistance,
    LexicalFrequencyProportion.name: LexicalFrequencyProportion,
    MorphologicalChoiceDivergence.name: MorphologicalChoiceDivergence,
    StereotypicalDivergence.name: StereotypicalDivergence,
    StereotypicalValueAttribution.name: StereotypicalValueAttribution,
    CounterfactualRobustness.name: CounterfactualRobustness,
    CounterfactualFairnessScore.name: CounterfactualFairnessScore,
    DemographicNextTokenProportion.name: DemographicNextTokenProportion,
    DemographicRepresentationDivergence.name: DemographicRepresentationDivergence,
    AccuracyDisparity.name: AccuracyDisparity,
    BiasAmplifierScore.name: BiasAmplifierScore,
    SensitiveNameSimilarity.name: SensitiveNameSimilarity,
    GradientBasedBiasEstimation.name: GradientBasedBiasEstimation,
    NaturalIndirectEffect.name: NaturalIndirectEffect,
    CooccurrenceAssociation.name: CooccurrenceAssociation,
    StereotypicalLogLikelihood.name: StereotypicalLogLikelihood,
}


def list_metrics():
    """Return sorted public metric names."""
    return sorted(METRIC_REGISTRY)


def get_metric(name: str) -> FairnessMetric:
    """Instantiate a metric by registry name."""
    try:
        cls = METRIC_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown metric {name!r}. Available: {', '.join(list_metrics())}"
        ) from exc
    return cls()


__all__ = [
    "AUL",
    "AULA",
    "AccuracyDisparity",
    "AllUnmaskedLikelihoodAttentionScore",
    "AllUnmaskedLikelihoodScore",
    "BiasAmplifierScore",
    "CAT",
    "CBS",
    "CEAT",
    "CPS",
    "CooccurrenceAssociation",
    "ContextAssociationTestScore",
    "ContextBasedDisparityScore",
    "ContrastBasedScore",
    "CounterfactualAucScore",
    "CounterfactualFairnessScore",
    "CounterfactualRobustness",
    "CrowSPairsScore",
    "DemographicNextTokenProportion",
    "DemographicRepresentationDivergence",
    "DisCo",
    "DiscoveryOfCorrelationsScore",
    "EqualOpportunityGap",
    "FairInferenceScore",
    "FairnessMetric",
    "GradientBasedBiasEstimation",
    "InferenceBiasScore",
    "LPBS",
    "LexicalFrequencyProportion",
    "LogProbabilityBiasScore",
    "METRIC_REGISTRY",
    "MetricResult",
    "MorphologicalChoiceDivergence",
    "NaturalIndirectEffect",
    "NormalizedPositionDistance",
    "PLL",
    "PseudoLogLikelihoodScore",
    "SEAT",
    "SensitiveNameSimilarity",
    "StereotypicalDivergence",
    "StereotypicalLogLikelihood",
    "StereotypicalValueAttribution",
    "TranslationSimilarityScore",
    "WEAT",
    "get_metric",
    "list_metrics",
]
