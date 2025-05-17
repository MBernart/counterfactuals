import numpy as np
import pandas as pd
import pickle
from .counterfactual import ClassLabel

class Dataset:
    """This is a helper class"""

    def __init__(
        self,
        name: str,
        data: pd.DataFrame,
        categorical_features: list[str],
        continuous_features: list[str],
        target: str,
        immutable_features: list[str],
        allowable_ranges: list[tuple[float, float]],
    ):
        self.name = name
        self.data = data
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.target = target
        self.immutable_features = immutable_features
        self.allowable_ranges = allowable_ranges

    def __iter__(self) -> Iterator[NDArray[Any]]:
        return iter(self.data)
        


class SerializableDataset(Dataset):
    def serialize(self) -> bytes:
        return pickle.dumps({
            'name': self.name,
            'data': self.data,
            'categorical_features': self.categorical_features,
            'continuous_features': self.continuous_features,
            'target': self.target,
            'immutable_features': self.immutable_features,
            'allowable_ranges': self.allowable_ranges,

        })

    @staticmethod
    def deserialize(data: bytes) -> Dataset:
        obj = pickle.loads(data)
        return SerializableDataset(
            obj['name'],
            obj['data'],
            obj['categorical_features'],
            obj['continuous_features'],
            obj['target'],
            obj['immutable_features'],
            obj['allowable_ranges'],
        )