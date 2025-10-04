# First start the redis message broker
# docker run -d -p 6379:6379 --name celery-redis redis

# Then start the explainers (you need to have properly configured python venv)
# celery -A explainers.wachter.main worker -l info -Q wachter,celery
# celery -A explainers.growing_spheres.main worker -l info -Q growing_spheres,celery

# If you prefer to use Docker, you can run
# docker build -t wachter-explainer -f explainers/wachter/Dockerfile .
# docker build -t growing-spheres-explainer -f explainers/growing_spheres/Dockerfile .
# docker run --rm -it --network host wachter-explainer
# docker run --rm -it --network host growing-spheres-explainer

# Alternatively you can pull the Docker images from our repository
# docker pull cfe.cs.put.poznan.pl/counterfactuals-wachter
# docker pull cfe.cs.put.poznan.pl/counterfactuals-growing-spheres

from explainers_lib.explainers.celery_remote import app
from explainers_lib.datasets import SerializableDataset
from sklearn.datasets import load_iris
from celery import group
from explainers_lib.counterfactual import Counterfactual
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

data = SerializableDataset(X_test, y_test, iris.feature_names, [], [], [], [])[:10]

# Loading the pretrained model
with open("temp_model.pt", "rb") as f:
    model_data = f.read()

# Ensemble
# TODO: Turn the ensemble example into a reusable class
explainers = list(reversed(app.send_task('ensemble.get_explainers').get()))
print(f"Available explainers: {explainers}")

collect_results_sig = app.signature('ensemble.collect_results')

# TODO: Need to check that selected explainers are actually available (ex. ping)

setup_chains = []
for explainer_name in explainers:

    set_dataset_sig = app.signature(
        f'{explainer_name}.set_dataset',
        args=[data.serialize()],
        queue=explainer_name
    )

    set_model_sig = app.signature(
        f'{explainer_name}.set_model',
        args=[model_data, 'torch'],
        queue=explainer_name
    )

    fit_sig = app.signature(
        f'{explainer_name}.fit',
        queue=explainer_name
    )

    setup_chain = group([set_dataset_sig, set_model_sig]) | fit_sig
    setup_chains.append(setup_chain)

setup_chord = group(setup_chains) | collect_results_sig
setup_results = setup_chord.apply_async().get()
print("All explainers setup complete.")

explain_tasks = []
for explainer_name in explainers:
    explain_tasks.append(
        app.signature(f'{explainer_name}.explain', queue=explainer_name)
    )
explain_chord = group(explain_tasks) | collect_results_sig
results = explain_chord.apply_async().get()

for result in results:
    print(f"Explainer: {result['explainer']}")

    if 'instance' in result:
        instance = SerializableDataset.deserialize(result['instance'])
        print(f"Instance: {instance.data}")
    else:
        print("Instance data not available.")

    if 'counterfactuals' in result and result['counterfactuals'] is not None:
        counterfactuals = [Counterfactual.deserialize(counterfactual) for counterfactual in result['counterfactuals']]
        print(f"Counterfactuals: {counterfactuals}")
    else:
        print("Counterfactuals not generated.")

# TODO: collect cfs and pass into an aggregator
