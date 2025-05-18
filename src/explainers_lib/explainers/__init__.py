from typing import Sequence
from ..counterfactual import Counterfactual
from ..datasets import Dataset
from ..model import Model

class Explainer:
    """This is an abstract class for an explainer"""

    def fit(self, model: Model, data: Dataset) -> None:
        """This method is used to fit the explainer"""
        raise NotImplementedError

    def explain(self, model: Model, data: Dataset) -> Sequence[Counterfactual]:
        """This method is used generate the counterfactuals"""
        raise NotImplementedError
