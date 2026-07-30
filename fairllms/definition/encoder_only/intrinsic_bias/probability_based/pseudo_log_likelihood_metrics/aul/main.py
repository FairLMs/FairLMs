"""Short demo of the public ``AllUnmaskedLikelihoodScore`` metric."""

from fairllms.datasets import CrowSPairs
from fairllms.metrics import AllUnmaskedLikelihoodScore
from fairllms.models import HuggingFaceModel


def main():
    model = HuggingFaceModel("bert-base-uncased", task="mlm")
    result = AllUnmaskedLikelihoodScore().compute(
        model=model,
        dataset=CrowSPairs(n_max=16),
    )
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
