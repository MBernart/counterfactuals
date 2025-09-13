from explainers_lib.model import TorchModel
from explainers_lib import SerializableDataset, Dataset, ClassLabel
import numpy as np
from pprint import pprint

with open("examples/temp_model.pt", "rb") as f:
    model_data = f.read()

model = TorchModel.deserialize(model_data)

data = SerializableDataset(
    np.random.rand(10, 4).astype(np.float32), None, [], [], [], [], []
)
pprint(model.predict_proba(data))
