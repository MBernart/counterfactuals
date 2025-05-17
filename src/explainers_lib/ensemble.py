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

    # probably want to explain single record at once
    def explain(self, data: Dataset, record_nr: int) -> list[Counterfactual]:
        """This method is used to generate counterfactuals"""

        all_counterfactuals = []
        for explainer in self.explainers:
            cfs = explainer.explain(self.model, data, record_nr) # must be implemented in explainers class
            all_counterfactuals.extend(cfs)
        
        filtered_counterfactuals = Aggregator(all_counterfactuals)
        return filtered_counterfactuals
