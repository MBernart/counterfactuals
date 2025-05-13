from .counterfactual import ClassLabel
from .datasets import Dataset
import pickle


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

    @staticmethod
    def deserialize(data: bytes) -> Model:
        raise NotImplementedError
    

# Example concrete implementation - similar should be added for different real models
class DummyModel(SerializableModel):
    """A trivial classifier that always returns class 0"""

    def __init__(self):
        self.trained = False

    def fit(self, data: Dataset) -> None:
        self.trained = True

    def predict(self, data: Dataset) -> list[ClassLabel]:
        if not self.trained:
            raise RuntimeError("Model not trained.")
        return [0 for _ in data]

    def serialize(self) -> bytes:
        return pickle.dumps({
            'trained': self.trained,
        })

    @staticmethod
    def deserialize(data: bytes) -> 'DummyModel':
        state = pickle.loads(data)
        model = DummyModel()
        model.trained = state['trained']
        return model
