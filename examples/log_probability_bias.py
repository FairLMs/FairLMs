"""Log Probability Bias Score via the public fairllms API."""

from fairllms.metrics import LogProbabilityBiasScore
from fairllms.models import HuggingFaceModel


def main():
    model = HuggingFaceModel("bert-base-uncased", task="mlm")
    result = LogProbabilityBiasScore().compute(
        model=model,
        attribute_words=["nurse", "surgeon", "teacher", "engineer"],
    )
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
