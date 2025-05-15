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
        self.model = model
        self.explainers = explainers
        self.aggregator = aggregator

    def fit(self, data: Dataset) -> None:
        """This method is used to train all explainers in the ensemble"""
        for explainer in self.explainers:
            explainer.fit(self.model, data)

    def explain(self, data: Dataset) -> list[Counterfactual]:
        """This method is used to generate counterfactuals"""
        cfs: list[Counterfactual] = []
        for explainer in self.explainers:
            cfs.extend(explainer.explain(self.model, data))
        return self.aggregator(cfs)
