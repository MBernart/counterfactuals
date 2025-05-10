from .counterfactual import ClassLabel
from .datasets import Dataset


class Model:
    """This is an abstract class for a black/white-box classifier"""

    def fit(self, data: Dataset) -> None:
        """This method is used to train the classifier"""
        raise NotImplementedError

    def predict(self, data: Dataset) -> list[ClassLabel]:
        """This method is used predict the class of instances"""
        raise NotImplementedError


class SerializableModel(Model):
    def serialize(self) -> bytes:
        raise NotImplementedError

    def deserialize(data: bytes) -> Model:
        raise NotImplementedError
