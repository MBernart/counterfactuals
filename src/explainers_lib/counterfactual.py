from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
import pickle

ClassLabel = int  # alias for a class label type

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