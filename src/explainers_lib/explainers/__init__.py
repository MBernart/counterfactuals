from typing import Sequence
from ..counterfactual import Counterfactual
from ..datasets import Dataset
from ..model import Model

class Explainer:
    """This is an abstract class for an explainer"""

    def fit(self) -> None:
        """This method is used to fit the explainer"""
        pass

    def explain(self) -> Sequence[Counterfactual]:
        """This method is used generate the counterfactuals"""
        pass
