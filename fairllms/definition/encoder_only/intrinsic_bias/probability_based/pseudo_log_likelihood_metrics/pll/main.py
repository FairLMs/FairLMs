"""Short demo of the public ``PseudoLogLikelihoodScore`` metric."""

from fairllms.datasets import CrowSPairs
from fairllms.metrics import PseudoLogLikelihoodScore
from fairllms.models import HuggingFaceModel


def main():
    model = HuggingFaceModel("bert-base-uncased", task="mlm")
    result = PseudoLogLikelihoodScore().compute(
        model=model,
        dataset=CrowSPairs(n_max=32),
    )
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
