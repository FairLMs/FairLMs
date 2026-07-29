"""Short demo of the public ``PseudoLogLikelihoodScore`` metric."""

from fairLLMs.datasets import CrowSPairs
from fairLLMs.metrics import PseudoLogLikelihoodScore
from fairLLMs.models import HuggingFaceModel


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
