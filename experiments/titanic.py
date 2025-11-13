import pandas as pd
from sklearn.model_selection import train_test_split
from explainers_lib.explainers.wachter import WachterExplainer
from explainers_lib.explainers.growing_spheres import GrowingSpheresExplainer
from explainers_lib.explainers.celery_explainer import ActionableRecourseExplainer
from explainers_lib.explainers.celery_explainer import FaceExplainer
from explainers_lib.explainers.celery_explainer import DiceExplainer
from explainers_lib.explainers.celery_explainer import AlibiCFProto
from explainers_lib.explainers.celery_explainer import AlibiCFRL
from explainers_lib.aggregators import Pareto, All
from explainers_lib.datasets import Dataset
from explainers_lib.ensemble import Ensemble
from explainers_lib.model import TorchModel


url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
df = pd.read_csv(url)
df = df.drop(['Name'], axis=1)
categorical_features = ['Sex', 'Pclass']
numerical_features = ['Age', 'Fare', 'Parents/Children Aboard', 'Siblings/Spouses Aboard']
target = 'Survived'


X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ds = Dataset(X_test, y_test.values, X_test.columns.tolist(), categorical_features=categorical_features, continuous_features=numerical_features)

print(ds.data)

input_dim = ds.data.shape[1]
with open("models/titanic_classifier.pt", "rb") as f:
    model_data = f.read()

model = TorchModel.deserialize(model_data)
print(model, input_dim)

ensemble = Ensemble(model, 
    [WachterExplainer(), GrowingSpheresExplainer(),                   # Local explainers
     ActionableRecourseExplainer(), DiceExplainer(), FaceExplainer(), # Carla explainers
     AlibiCFProto(), AlibiCFRL()],                                    # Alibi explainers
    All())
ensemble.fit(ds)
cfs = ensemble.explain(ds[:5], pretty_print=True, pretty_print_postprocess=ds.inverse_transform, feature_names=ds.features)
cfs

