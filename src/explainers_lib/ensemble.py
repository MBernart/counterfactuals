from .model import Model
from .explainers import Explainer
from .aggregators import Aggregator
from .counterfactual import Counterfactual
from .datasets import Dataset


class Ensemble:
    def __init__(
        self, model: Model, explainers: list[Explainer], aggregator: Aggregator
    ):
        """Constructs an ensemble"""
        raise NotImplementedError

    def fit(self, data: Dataset) -> None:
        """This method is used to train all explainers in the ensemble"""
        raise NotImplementedError

    def explain(self, data: Dataset) -> list[Counterfactual]:
        """This method is used to generate counterfactuals"""
        raise NotImplementedError
