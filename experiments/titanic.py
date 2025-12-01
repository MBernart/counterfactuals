import pandas as pd
from sklearn.model_selection import train_test_split
from explainers_lib.explainers.native.wachter import WachterExplainer
from explainers_lib.explainers.native.growing_spheres import GrowingSpheresExplainer
from explainers_lib.explainers.celery_explainer import ActionableRecourseExplainer
from explainers_lib.explainers.native.face import FaceExplainer
from explainers_lib.explainers.dice.dice import DiceExplainer
from explainers_lib.explainers.celery_explainer import AlibiCFProto
from explainers_lib.explainers.celery_explainer import AlibiCFRL
from explainers_lib.aggregators import Pareto, IdealPoint, BalancedPoint, TOPSIS, DensityBased, ScoreBasedAggregator
from explainers_lib.datasets import Dataset
from explainers_lib.ensemble import Ensemble, cfs_group_by_original_data, print_cfs
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

ensemble = Ensemble(
    model, 
    [
        # Native
        WachterExplainer(lambda_param=[0.1, 0.5, 1, 5, 10, 50, 100]),
        GrowingSpheresExplainer(max_radius=10),
        FaceExplainer(fraction=1.0),
        # # Carla
        # # TODO(patryk): currently broken, but I am working on it! 
        # # ActionableRecourseExplainer(),
        # # Dice
        DiceExplainer(num_cfs=10),
        # # Alibi
        AlibiCFProto(),
        AlibiCFRL()
    ])
explainers = ensemble.get_explainers_repr()
ensemble.fit(ds)

explain_idx = 20
all_cfs = ensemble.explain(ds[explain_idx], pretty_print=True, pretty_print_postprocess=ds.inverse_transform, feature_names=ds.features)

for selector in [DensityBased(), Pareto(), IdealPoint(), BalancedPoint(), TOPSIS()]:
    if isinstance(selector, ScoreBasedAggregator):
        selector.fit(model, ds)

    all_selected_cfs = list()
    for cfs in cfs_group_by_original_data(all_cfs).values():
        selected_cfs = selector(cfs)
        all_selected_cfs.extend(selected_cfs)

    print(selector)
    print_cfs(all_selected_cfs, ds.features, ds[explain_idx], explainers, model, ds.inverse_transform)
