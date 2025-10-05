# You can run the explainer locally
from explainers_lib.explainers.wachter import WachterExplainer

# Or you can run it via celery
# from explainers_lib.explainers.celery_explainer import WachterExplainer

# To do this, start the redis message broker
# docker run -d -p 6379:6379 --name celery-redis redis

# Then start the explainer (you need to have properly configured python venv)
# celery -A explainers.wachter.main worker -l info -n wachter_worker@%h -Q wachter,celery

# If you prefer to use Docker, you can pull the image from our repository
# docker pull cfe.cs.put.poznan.pl/counterfactuals-wachter

# Alternatively you can build and run the image
# docker build -t wachter-explainer -f explainers/wachter/Dockerfile .
# docker run --rm -it --network host wachter-explainer

from explainers_lib import TorchModel, Dataset
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

iris = load_iris()

with open("temp_model.pt", "rb") as f:
    model_data = f.read()

model = TorchModel.deserialize(model_data)

# Load the Iris dataset and create identical SS as the model was trained on
# literally copy-pasted from training script
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data["species"] = iris.target

label_encoder = LabelEncoder()
data["species"] = label_encoder.fit_transform(data["species"])

X = data.drop("species", axis=1).values
y = data["species"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

data = Dataset(X_test, y_test, iris.feature_names, [], [], [], [])

explainer = WachterExplainer()
explainer.fit(model, data)
cfs = explainer.explain(model, data[:5])

print(f"Wachter generated counterfactuals: {cfs}")
