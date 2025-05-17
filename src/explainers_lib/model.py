from typing import Dict, List
from .counterfactual import ClassLabel
from .datasets import Dataset
import tensorflow as tf
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

    def predict(self) -> list[ClassLabel]:
        """This method is used predict the class of instances"""
        pass
        # raise NotImplementedError



class SerializableModel(Model):
    def serialize(self) -> bytes:
        pass
        # raise NotImplementedError

    @staticmethod
    def deserialize(self) -> Model:
        pass
        # raise NotImplementedError
    


class TFModel(SerializableModel):
    def __init__(self, model: tf.keras.Model, data: pd.DataFrame, columns_ohe_order: List[str]) -> None:
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

        if isinstance(x, tf.Variable):
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
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
    def predict_proba(self, x):
        x = self._prepare_input(x)
        return self._mymodel.predict(x)
    
    def serialize(self) -> bytes:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.h5')
            self._mymodel.save(path)

            with open(path, 'rb') as f:
                model_bytes = f.read()

        return pickle.dumps({
            'model_bytes': model_bytes,
            'columns_order': self.columns_order
        })

    @staticmethod
    def deserialize(data: bytes) -> 'TFModel':
        state = pickle.loads(data)
        model_bytes = state['model_bytes']
        columns_order = state['columns_order']

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.h5')
            with open(path, 'wb') as f:
                f.write(model_bytes)

            model = tf.keras.models.load_model(path)

        return TFModel(model=model, data=None, columns_ohe_order=columns_order)