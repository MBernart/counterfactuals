import io
from typing import Optional, List, Tuple
import numpy as np
import pickle
from .counterfactual import ClassLabel


class Dataset:
    """This is a helper class"""

    def __init__(
        self,
        data: np.ndarray,
        target: List[ClassLabel],
        features: List[str],
        categorical_features: List[str],
        continuous_features: List[str],
        immutable_features: List[str],
        allowable_ranges: List[Tuple[float, float]],
    ):
        self.data = data
        self.target = target.tolist() if isinstance(target, np.ndarray) else target
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

    def like(self, data: np.ndarray, target: Optional[np.ndarray] = None) -> "Dataset":
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
        return pickle.dumps(
            {
                "data": Dataset._array_to_bytes(self.data),
                "target": self.target,
                "features": self.features,
                "categorical_features": self.categorical_features,
                "continuous_features": self.continuous_features,
                "immutable_features": self.immutable_features,
                "allowable_ranges": self.allowable_ranges,
            },
            protocol=4
        )

    @staticmethod
    def deserialize(data: bytes) -> "Dataset":
        obj = pickle.loads(data)
        return Dataset(
            Dataset._bytes_to_array(obj["data"]),
            obj["target"],
            obj["features"],
            obj["categorical_features"],
            obj["continuous_features"],
            obj["immutable_features"],
            obj["allowable_ranges"],
        )
