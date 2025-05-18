from explainers_lib import TorchModel, SerializableDataset, Ensemble, Pareto
from explainers_lib.explainers.remote import RemoteExplainer
from sklearn.datasets import load_iris

# Loading the data
iris = load_iris()
data = SerializableDataset(iris.data, iris.target, iris.feature_names, [], [], [], [])
data = data[0:2]

# Load the black box model
with open("temp_model.pt", "rb") as f:
    model_data = f.read()

model = TorchModel.deserialize(model_data)

# Define the ensemble
explainer = RemoteExplainer("localhost", 8000)
ensemble = Ensemble(model=model, explainers=[explainer], aggregator=Pareto())

# Train the ensemble
ensemble.fit(data)

# Test the ensemble
ensemble.explain(data[0])
