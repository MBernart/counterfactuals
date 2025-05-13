import numpy as np
import pickle


class Dataset:
    """This is a helper class"""

    def __init__(
        self,
        data: np.ndarray,
        features: list[str],
        immutable_features: list[str],
        categorical_features: list[str],
        allowable_ranges: list[tuple[float, float]],
    ):
        self.data = data
        self.features = features
        self.immutable_features = immutable_features
        self.categorical_features = categorical_features
        self.allowable_ranges = allowable_ranges

    def __iter__(self):
        return iter(self.data)


class SerializableDataset(Dataset):
    def serialize(self) -> bytes:
        return pickle.dumps({
            'data': self.data,
            'features': self.features,
            'immutable_features': self.immutable_features,
            'categorical_features': self.categorical_features,
            'allowable_ranges': self.allowable_ranges,
        })

    @staticmethod
    def deserialize(data: bytes) -> Dataset:
        obj = pickle.loads(data)
        return SerializableDataset(
            obj['data'],
            obj['features'],
            obj['immutable_features'],
            obj['categorical_features'],
            obj['allowable_ranges'],
        )