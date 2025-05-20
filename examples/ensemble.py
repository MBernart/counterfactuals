from explainers_lib import Ensemble, TorchModel, Pareto, SerializableDataset as Dataset
from explainers_lib.explainers.growing_spheres import GrowingSpheresExplainer
from explainers_lib.explainers.remote import RemoteExplainerFactory
from sklearn.datasets import load_iris

# Loading the data
iris = load_iris()
data = Dataset(iris.data, iris.target, iris.feature_names, [], iris.feature_names, [], [])

# Load the black box model
with open("temp_model.pt", "rb") as f:
    model_data = f.read()

model = TorchModel.deserialize(model_data)

# Define the ensemble
explainer = GrowingSpheresExplainer()
ensemble = Ensemble(model=model, explainers=[explainer] * 3, aggregator=Pareto())

# Train the ensemble
ensemble.fit(data)

# Test the ensemble
ensemble.explain(data[0], data)
