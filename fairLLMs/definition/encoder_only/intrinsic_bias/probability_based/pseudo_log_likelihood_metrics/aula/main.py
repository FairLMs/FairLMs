"""Short demo of the public ``AllUnmaskedLikelihoodAttentionScore`` metric."""

from fairLLMs.datasets import CrowSPairs
from fairLLMs.metrics import AllUnmaskedLikelihoodAttentionScore
from fairLLMs.models import HuggingFaceModel


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
