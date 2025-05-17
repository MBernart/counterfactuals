from typing import Sequence
from .counterfactual import Counterfactual
from .datasets import Dataset, SerializableDataset
from .model import Model, SerializableModel


class Explainer:
    """This is an abstract class for an explainer"""

    def __init__(self):
        pass

    def fit(self) -> None:
        """This method is used to fit the explainer"""
        pass
        # raise NotImplementedError

    def explain(self) -> Sequence[Counterfactual]:
        """This method is used generate the counterfactuals"""
        pass
        # raise NotImplementedError


class RemoteExplainer:
    def connect(self, ip: str) -> None:
        raise NotImplementedError

    def fit(self, model: SerializableModel, data: SerializableDataset) -> None:
        """This method is used to fit the explainer"""
        raise NotImplementedError

    def explain(
        self, model: SerializableModel, data: SerializableDataset, record_nr: int
    ) -> Sequence[Counterfactual]:
        """This method is used generate the counterfactuals"""
        raise NotImplementedError
