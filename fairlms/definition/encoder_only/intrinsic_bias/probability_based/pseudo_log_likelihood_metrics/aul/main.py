"""Short demo of the public ``AllUnmaskedLikelihoodScore`` metric."""

from fairlms.datasets import CrowSPairs
from fairlms.metrics import AllUnmaskedLikelihoodScore
from fairlms.models import HuggingFaceModel


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
