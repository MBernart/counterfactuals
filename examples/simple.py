from explainers_lib import Ensemble, DummyModel as Model, MainServerFactory, Pareto, SerializableDataset as Dataset
from sklearn.datasets import load_iris

# Loading the data
iris = load_iris()
data = Dataset(iris.data, iris.target, iris.feature_names, [], [], [])

# Create the black box model
model = Model()

# Define the ensemble
# explainer = RemoteExplainer("localhost:8000")
# ensemble = Ensemble(model=model, explainers=[explainer], aggregator=Pareto())

# # Train the ensemble
# ensemble.fit(data)

# # Test the ensemble
# alternative_0 = Dataset(iris.data[0], iris.target[0], iris.feature_names, [], [], []) # could be replaced with data.take(1)
# ensemble.explain(alternative_0)

from twisted.internet import reactor, protocol, defer, task
from twisted.protocols.basic import LineReceiver
import pickle
import sys
import time

reactor.connectTCP("localhost", 8000, MainServerFactory(data, model))
reactor.run()
