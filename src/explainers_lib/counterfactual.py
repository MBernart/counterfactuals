from dataclasses import dataclass
import io
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

    @staticmethod
    def _array_to_bytes(arr: np.ndarray) -> bytes:
        """Helper to serialize a single np.ndarray to bytes."""
        with io.BytesIO() as f:
            np.save(f, arr)
            return f.getvalue()

    @staticmethod
    def _bytes_to_array(b: bytes) -> np.ndarray:
        """Helper to deserialize bytes back into a single np.ndarray."""
        with io.BytesIO(b) as f:
            return np.load(f, allow_pickle=False)

    def serialize(self) -> bytes:
        return pickle.dumps({
            "original_data": Counterfactual._array_to_bytes(self.original_data),
            "data": Counterfactual._array_to_bytes(self.data),
            "original_class": self.original_class,
            "target_class": self.target_class,
            "explainer": self.explainer,
        }, protocol=4)

    @staticmethod
    def deserialize(data: bytes) -> "Counterfactual":
        state = pickle.loads(data)
        return Counterfactual(
            original_data = Counterfactual._bytes_to_array(state["original_data"]),
            data = Counterfactual._bytes_to_array(state["data"]),
            original_class = state["original_class"],
            target_class = state["target_class"],
            explainer = state["explainer"]
        )