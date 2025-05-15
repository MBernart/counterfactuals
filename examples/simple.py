from explainers_lib import Ensemble, Model, Explainer, Pareto, Dataset
from sklearn.datasets import load_iris

# Loading the data
iris = load_iris()
data = Dataset(iris.data, iris.target, iris.feature_names, [], [], [])

# Create the black box model
model = Model()

# Define the ensemble
explainer = Explainer()
ensemble = Ensemble(model=model, explainers=[explainer], aggregator=Pareto())

# Train the ensemble
ensemble.fit(data)

# Test the ensemble
alternative_0 = Dataset(iris.data[0], iris.target[0], iris.feature_names, [], [], []) # could be replaced with data.take(1)
ensemble.explain(alternative_0)
