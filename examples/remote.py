from explainers_lib import TorchModel, Dataset
from explainers_lib.explainers.remote import RemoteExplainerFactory
from sklearn.datasets import load_iris

# Loading the data
iris = load_iris()
data = Dataset(iris.data, iris.target, iris.feature_names, [], [], [], [])
data = data[0:2]

# Load the black box model
with open("temp_model.pt", "rb") as f:
    model_data = f.read()

model = TorchModel.deserialize(model_data)

# Run a remote explainer
from twisted.internet import reactor
reactor.connectTCP("localhost", 8000, RemoteExplainerFactory(data, model))
reactor.run()
