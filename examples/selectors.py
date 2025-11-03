from explainers_lib.explainers.wachter import WachterExplainer
from explainers_lib.explainers.growing_spheres import GrowingSpheresExplainer
from explainers_lib.explainers.celery_explainer import ActionableRecourseExplainer
from explainers_lib.explainers.celery_explainer import FaceExplainer
from explainers_lib.explainers.celery_explainer import DiceExplainer
from explainers_lib.explainers.celery_explainer import AlibiCFProto
from explainers_lib.explainers.celery_explainer import AlibiCFRL
from explainers_lib.aggregators import Pareto, IdealPoint, BalancedPoint, TOPSIS, DensityBased, ScoreBasedAggregator
from explainers_lib.datasets import Dataset
from explainers_lib.ensemble import Ensemble, print_cfs
from explainers_lib.model import TorchModel
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split

# Dataset preparation
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

data = Dataset(X_test, y_test, iris.feature_names, [], iris.feature_names, [], [])

# Loading the pretrained model
with open("temp_model.pt", "rb") as f:
    model_data = f.read()

model = TorchModel.deserialize(model_data)

# Ensemble
ensemble = Ensemble(
    model,
    [WachterExplainer(), GrowingSpheresExplainer(),                   # Local explainers
     ActionableRecourseExplainer(), DiceExplainer(), FaceExplainer(), # Carla explainers
     AlibiCFProto(), AlibiCFRL()])                                    # Alibi explainers
explainers = ensemble.get_explainers_repr()
print(f"Used explainers: {[explainer for explainer in explainers]}")

ensemble.fit(data)
print(f"Ensemble fitting complete")

all_cfs = ensemble.explain(data[:5])
print("All cfs")
print_cfs(all_cfs, data.features, data[:5], explainers, model, scaler.inverse_transform, label_encoder.inverse_transform)

for selector in [Pareto(), IdealPoint(), BalancedPoint(), TOPSIS(), DensityBased()]:
    if isinstance(selector, ScoreBasedAggregator):
        selector.fit(model, data)

    selected = selector(all_cfs)
    print(selector)
    print_cfs(selected, data.features, data[:5], explainers, model, scaler.inverse_transform, label_encoder.inverse_transform)
