from typing import Sequence
from .counterfactual import Counterfactual
from .datasets import Dataset, SerializableDataset
from .model import Model, SerializableModel
import numpy as np
import torch


class Explainer:
    """This is an abstract class for an explainer"""

    def fit(self, model: Model, data: Dataset) -> None:
        """This method is used to fit the explainer"""
        raise NotImplementedError

    def explain(self, model: Model, data: Dataset) -> Sequence[Counterfactual]:
        """This method is used generate the counterfactuals"""
        raise NotImplementedError


class RemoteExplainer:
    def connect(ip: str) -> None:
        raise NotImplementedError

    def fit(self, model: SerializableModel, data: SerializableDataset) -> None:
        """This method is used to fit the explainer"""
        raise NotImplementedError

    def explain(
        self, model: SerializableModel, data: SerializableDataset
    ) -> Sequence[Counterfactual]:
        """This method is used generate the counterfactuals"""
        raise NotImplementedError
