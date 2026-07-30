"""Short demo of the public ``AllUnmaskedLikelihoodAttentionScore`` metric."""

from fairllms.datasets import CrowSPairs
from fairllms.metrics import AllUnmaskedLikelihoodAttentionScore
from fairllms.models import HuggingFaceModel


def main():
    model = HuggingFaceModel("bert-base-uncased", task="mlm")
    result = AllUnmaskedLikelihoodAttentionScore().compute(
        model=model,
        dataset=CrowSPairs(n_max=16),
    )
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
