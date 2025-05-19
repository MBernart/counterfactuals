from typing import Any, Iterator, Self
import numpy as np
from numpy.typing import NDArray
import pickle
from .counterfactual import ClassLabel

class Dataset:
    """This is a helper class"""

    def __init__(
        self,
        data: NDArray[Any],
        target: list[ClassLabel],
        features: list[str],
        categorical_features: list[str],
        continuous_features: list[str],
        immutable_features: list[str],
        allowable_ranges: list[tuple[float, float]],
    ):
        self.data = data
        self.target = target
        self.features = features
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.immutable_features = immutable_features
        self.allowable_ranges = allowable_ranges

        self.categorical_features_ids = [features.index(f) for f in categorical_features]
        self.continuous_features_ids = [features.index(f) for f in continuous_features]
        self.immutable_features_ids = [features.index(f) for f in immutable_features]

    class DatasetIterator:
        def __init__(self, dataset: 'Dataset'):
            self.dataset = dataset
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.index >= len(self.dataset.data):
                raise StopIteration
            result = self.dataset[self.index]
            self.index += 1
            return result

    def __iter__(self) -> DatasetIterator:
        return Dataset.DatasetIterator(self)

    def __getitem__(self, key) -> Self:
        if isinstance(key, slice):
            data = self.data[key.start:key.stop:key.step]
        elif isinstance(key, int):
            data = self.data[key:key+1]
        else:
            raise TypeError("Invalid argument type.")
        return self.like(data)
    
    def like(self, data: NDArray[Any]) -> Self:
        return self.__class__(data, self.target, self.features, self.categorical_features, self.continuous_features, self.immutable_features, self.allowable_ranges)


class SerializableDataset(Dataset):
    def serialize(self) -> bytes:
        return pickle.dumps({
            'data': self.data,
            'target': self.target,
            'features': self.features,
            'categorical_features': self.categorical_features,
            'continuous_features': self.continuous_features,
            'immutable_features': self.immutable_features,
            'allowable_ranges': self.allowable_ranges,
            'categorical_features_ids': self.categorical_features_ids,
            'continuous_features_ids': self.continuous_features_ids,
            'immutable_features_ids': self.immutable_features_ids
        })

    @staticmethod
    def deserialize(data: bytes) -> Self:
        obj = pickle.loads(data)
        return SerializableDataset(
            obj['data'],
            obj['target'],
            obj['features'],
            obj['categorical_features'],
            obj['continuous_features'],
            obj['immutable_features'],
            obj['allowable_ranges'],
            obj['categorical_features_ids'],
            obj['continuous_features_ids'],
            obj['immutable_features_ids'],
        )