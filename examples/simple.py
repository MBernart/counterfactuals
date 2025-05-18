from explainers_lib import Ensemble, TorchModel, RemoteExplainerFactory, Pareto, SerializableDataset as Dataset
from sklearn.datasets import load_iris

# Loading the data
iris = load_iris()
data = Dataset(iris.data[0:2], iris.target, iris.feature_names, [], [], [])

# Load the black box model
with open("temp_model.pt", "rb") as f:
    model_data = f.read()

model = TorchModel.deserialize(model_data)

# Define the ensemble
# explainer = RemoteExplainer("localhost", 8000)
# ensemble = Ensemble(model=model, explainers=[explainer], aggregator=Pareto())

# # Train the ensemble
# ensemble.fit(data)

# # Test the ensemble
# alternative_0 = Dataset(iris.data[0], iris.target[0], iris.feature_names, [], [], []) # could be replaced with data.take(1)
# ensemble.explain(alternative_0)

from twisted.internet import reactor
reactor.connectTCP("localhost", 8000, RemoteExplainerFactory(data, model))
reactor.run()
