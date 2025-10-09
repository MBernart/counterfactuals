from dataclasses import dataclass
import numpy as np
import pickle

# TODO: dynamic import based on python version
# from typing import TypeAlias
from typing_extensions import TypeAlias

ClassLabel: TypeAlias = int  # alias for a class label type

# For now not sure whether it should be in a separate directory
@dataclass
class Counterfactual:
    """This is a helper class"""

    original_data: np.ndarray
    data: np.ndarray
    original_class: ClassLabel
    target_class: ClassLabel
    explainer: str

    def serialize(self) -> bytes:
        return pickle.dumps(self)

    @staticmethod
    def deserialize(data: bytes) -> "Counterfactual":
        return pickle.loads(data)