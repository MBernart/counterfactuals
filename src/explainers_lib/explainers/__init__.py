from ..counterfactual import Counterfactual
from ..datasets import Dataset
from ..model import Model
from typing import List

class Explainer:
    """This is an abstract class for an explainer"""

    def fit(self, model: Model, data: Dataset) -> None:
        """This method is used to fit the explainer"""
        pass

    def explain(self, model: Model, data: Dataset) -> List[Counterfactual]:
        """This method is used generate the counterfactuals"""
        pass

    def __repr__(self) -> str:
        """This method is used to represent the explainer"""
        pass
