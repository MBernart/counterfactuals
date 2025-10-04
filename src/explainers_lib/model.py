import numpy as np
from numpy.typing import NDArray
from typing import List, Any
from .counterfactual import ClassLabel
from .datasets import Dataset
import pandas as pd
import tempfile
import os
import pickle


class Model:
    """This is an abstract class for a black/white-box classifier"""

    def __init__(self):
        pass

    def fit(self) -> None:
        """This method is used to fit the model"""
        pass
        # raise NotImplementedError

    def predict(self, data: Dataset) -> list[ClassLabel]:
        """This method is used predict the class of instances"""
        pass
        # raise NotImplementedError

    def predict_proba(self, x) -> np.ndarray:
        """This method is used to predict the class probabilities of instances"""
        pass
        # raise NotImplementedError

    def serialize(self) -> tuple[bytes, str]:
        pass
        # raise NotImplementedError

    @staticmethod
    def deserialize(data: bytes, type: str) -> "Model":
        if type == "tensorflow":
            return TFModel.deserialize(data)
        elif type == "torch":
            return TorchModel.deserialize(data)
        raise RuntimeError(f"Unknown model type: {type}")

class TFModel(Model):
    def __init__(self, model: 'tf.keras.Model', data: pd.DataFrame, columns_ohe_order: List[str]) -> None:
        import tensorflow as tf
        self._tf = tf
        self._mymodel = model#self.__load_model()
        self.data = data
        self.columns_order = columns_ohe_order
    
    def __call__(self, data):
        return self._mymodel(data)

    # List of the feature order the ml model was trained on
    @property
    def feature_input_order(self):
        return self.columns_order

    # The ML framework the model was trained on
    @property
    def backend(self):
        return "tensorflow"

    # The black-box model object
    @property
    def raw_model(self):
        return self._mymodel

    def _prepare_input(self, x):
        if isinstance(x, pd.DataFrame):
            x = x[self.feature_input_order].to_numpy()

        if isinstance(x, self._tf.Variable):
            with self._tf.compat.v1.Session() as sess:
                sess.run(self._tf.compat.v1.global_variables_initializer())
                x = x.eval(session=sess)

        return x

    # The predict function outputs
    # the continuous prediction of the model
    def predict(self, x):
        x = self._prepare_input(x)
        return self._mymodel.predict(x)
    
    # @tf.function(experimental_relax_shapes=True)
    # def predictTensor(self, x):
    #     self._mymodel.predict(x, steps=1)

    # The predict_proba method outputs
    # the prediction as class probabilities
    def predict_proba(self, x) -> np.ndarray:
        x = self._prepare_input(x)
        return self._mymodel.predict(x)
    
    def serialize(self) -> tuple[bytes, str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.h5')
            self._mymodel.save(path)

            with open(path, 'rb') as f:
                model_bytes = f.read()

        return (
            pickle.dumps({
                'model_bytes': model_bytes,
                'columns_order': self.columns_order
            }), 
            "tensorflow"
        )

    @staticmethod
    def deserialize(data: bytes) -> 'TFModel':
        import tensorflow as tf
        state = pickle.loads(data)
        model_bytes = state['model_bytes']
        columns_order = state['columns_order']

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.h5')
            with open(path, 'wb') as f:
                f.write(model_bytes)

            model = tf.keras.models.load_model(path)

        return TFModel(model=model, data=None, columns_ohe_order=columns_order)


class TorchModel(Model):
    def __init__(self, model):
        self._model = model
        import torch
        self._torch = torch

    def fit(self, data: Dataset) -> None:
        raise NotImplementedError

    def predict(self, data: Dataset) -> list[ClassLabel]:
        labels = []
        for instance in data.data:
            # Convert the instance to a PyTorch Tensor
            instance_tensor = self._torch.tensor(instance, dtype=self._torch.float32)

            # Move the tensor to the same device as the model (if using CUDA)
            instance_tensor = instance_tensor.to(
                "cuda" if next(self._model.parameters()).is_cuda else "cpu"
            )  # model.device should be 'cpu' or 'cuda'

            # Get the predicted class by passing the instance through the model
            with self._torch.no_grad():  # Don't compute gradients during inference
                preds = self._model(instance_tensor.unsqueeze(0))  # Add batch dimension
                labels.append(int(self._torch.argmax(preds)))
        return np.array(labels)

    def predict_proba(self, data: NDArray[Any]) -> np.ndarray:
        data = self._torch.tensor(data.data, dtype=self._torch.float32)
        data = data.to("cuda" if next(self._model.parameters()).is_cuda else "cpu")

        with self._torch.no_grad():
            logits = self._model(data)
            probabilities = self._torch.nn.functional.softmax(logits, dim=1)

        return probabilities.cpu().numpy()

    def serialize(self) -> tuple[bytes, str]:
        import io

        buffer = io.BytesIO()

        self._torch.jit.save(self._model, buffer)

        buffer.seek(0)
        return (buffer.read(), "torch")

    @staticmethod
    def deserialize(data: bytes) -> Model:
        import torch
        import io

        buffer = io.BytesIO(data)

        return TorchModel(torch.jit.load(buffer))
