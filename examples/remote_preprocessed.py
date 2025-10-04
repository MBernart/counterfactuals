from explainers_lib import TorchModel, Dataset
from explainers_lib.explainers.celery_explainer import CeleryExplainer
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

explainer = CeleryExplainer("wachter")
explainer.fit(model, data)
cfs = explainer.explain(model, data[:5])

print(f"Wachter generated counterfactuals: {cfs}")
