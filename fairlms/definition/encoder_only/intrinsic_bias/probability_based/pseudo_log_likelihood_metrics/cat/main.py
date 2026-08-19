"""Short demo of the public ``ContextAssociationTestScore`` metric."""

from fairlms.datasets import StereoSet
from fairlms.metrics import ContextAssociationTestScore
from fairlms.models import HuggingFaceModel


def main():
    model = HuggingFaceModel("bert-base-uncased", task="mlm")
    dataset = StereoSet(config="intrasentence", as_triples=True, n_max=16)
    result = ContextAssociationTestScore().compute(model=model, dataset=dataset)
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
