from typing import Any, Iterator, Optional, List, Tuple
import numpy as np
from numpy.typing import NDArray
import pickle


class Dataset:
    """This is a helper class"""

    def __init__(
        self,
        data: NDArray[Any],
        target: List[int],
        features: List[str],
        categorical_features: List[str],
        continuous_features: List[str],
        immutable_features: List[str],
        allowable_ranges: List[Tuple[float, float]],
    ):
        self.data = data
        self.target = target
        self.features = features
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.immutable_features = immutable_features
        self.allowable_ranges = allowable_ranges

        self.categorical_features_ids = [
            features.index(f) for f in categorical_features
        ]
        self.continuous_features_ids = [features.index(f) for f in continuous_features]
        self.immutable_features_ids = [features.index(f) for f in immutable_features]

    class DatasetIterator:
        def __init__(self, dataset: "Dataset"):
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

    def __getitem__(self, key) -> "Dataset":
        if isinstance(key, slice):
            data = self.data[key.start : key.stop : key.step]
            target = (
                self.target[key.start : key.stop : key.step]
                if self.target is not None
                else None
            )
        elif isinstance(key, int):
            data = self.data[key : key + 1]
            target = self.target[key : key + 1] if self.target is not None else None
        else:
            raise TypeError("Invalid argument type.")
        return self.like(data, target)

    def like(self, data: NDArray[Any], target: Optional[NDArray[Any]] = None) -> "Dataset":
        if target is None:
            target = self.target
        return self.__class__(
            data,
            target,
            self.features,
            self.categorical_features,
            self.continuous_features,
            self.immutable_features,
            self.allowable_ranges,
        )

    def serialize(self) -> bytes:
        return pickle.dumps(
            {
                "data": self.data,
                "target": self.target,
                "features": self.features,
                "categorical_features": self.categorical_features,
                "continuous_features": self.continuous_features,
                "immutable_features": self.immutable_features,
                "allowable_ranges": self.allowable_ranges,
            }
        )

    @staticmethod
    def deserialize(data: bytes) -> "Dataset":
        obj = pickle.loads(data)
        return Dataset(
            obj["data"],
            obj["target"],
            obj["features"],
            obj["categorical_features"],
            obj["continuous_features"],
            obj["immutable_features"],
            obj["allowable_ranges"],
        )
