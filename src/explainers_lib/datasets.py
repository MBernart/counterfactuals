import numpy as np


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
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


class SerializableDataset(Dataset):
    def serialize(self) -> bytes:
        raise NotImplementedError

    def deserialize(data: bytes) -> Dataset:
        raise NotImplementedError
