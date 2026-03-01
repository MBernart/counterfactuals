# You can run the explainers locally
from explainers_lib.explainers.native.wachter import WachterExplainer
from explainers_lib.explainers.native.growing_spheres import GrowingSpheresExplainer

# Or you can run it via celery, or even run some locally and some via celery
# from explainers_lib.explainers.celery_explainer import WachterExplainer
# from explainers_lib.explainers.celery_explainer import GrowingSpheresExplainer
from explainers_lib.explainers.celery_explainer import ActionableRecourseExplainer
from explainers_lib.explainers.celery_explainer import FaceExplainer
from explainers_lib.explainers.celery_explainer import DiceExplainer
from explainers_lib.explainers.celery_explainer import AlibiCFProto
from explainers_lib.explainers.celery_explainer import AlibiCFRL

# To do this, start the redis message broker
# docker run -d -p 6379:6379 --name celery-redis redis

# Then start the explainers (you need to have properly configured python venv)
# celery -A explainers.wachter.main worker -l info -n wachter_worker@%h -Q wachter,celery
# celery -A explainers.growing_spheres.main worker -l info -n growing_spheres_worker@%h -Q growing_spheres,celery

# If you prefer to use Docker, you can pull the images from our repository
# docker pull cfe.cs.put.poznan.pl/counterfactuals-wachter
# docker pull cfe.cs.put.poznan.pl/counterfactuals-growing-spheres

# Alternatively you can build and run the images
# docker build -t wachter-explainer -f explainers/wachter/Dockerfile .
# docker build -t growing-spheres-explainer -f explainers/growing_spheres/Dockerfile .
# docker run --rm -it --network host wachter-explainer
# docker run --rm -it --network host growing-spheres-explainer

from explainers_lib.aggregators import Pareto
from explainers_lib.datasets import Dataset
from explainers_lib.ensemble import Ensemble
from explainers_lib.model import TorchModel
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split

# Dataset preparation
iris = load_iris()

data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data["species"] = iris.target

label_encoder = LabelEncoder()
data["species"] = label_encoder.fit_transform(data["species"])

X = data.drop("species", axis=1)
y = data["species"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

data = Dataset(X_test, y_test, continuous_features=iris.feature_names)

# Loading the pretrained model
with open("temp_model.pt", "rb") as f:
    model_data = f.read()

model = TorchModel.deserialize(model_data)

# Ensemble
ensemble = Ensemble(
    model,
    [
        # Native
        WachterExplainer(),
        GrowingSpheresExplainer(),
        # FaceExplainer(),
        # Carla
        # TODO(patryk): currently broken, but I am working on it! 
        # ActionableRecourseExplainer(),
        # Dice
        # DiceExplainer(),
        # Alibi
        # AlibiCFProto(),
        # AlibiCFRL()
    ],
    Pareto())
print(f"Used celery explainers: {[explainer.explainer_name for explainer in ensemble.celery_explainers]}")

ensemble.fit(data)
print(f"Ensemble fitting complete")

cfs = ensemble.explain(data[:5],
                       include_scores=True,
                       pretty_print=True,
                       pretty_print_postprocess=data.inverse_transform,
                       pretty_print_postprocess_target=label_encoder.inverse_transform)
print(f"Number of generated cfs: {len(cfs)}")

# Print scores table
from rich.console import Console
from rich.table import Table

console = Console()
table = Table(title="Counterfactual Scores")

table.add_column("Explainer", justify="left", style="cyan")
table.add_column("Target Class", justify="center", style="magenta")
table.add_column("Proximity", justify="right", style="green")
table.add_column("Feasibility", justify="right", style="green")
table.add_column("Discriminative Power", justify="right", style="green")

for cf in cfs:
    scores = cf.metadata.get("scores", {})
    # Note: Column names in scores depend on k_neighbors params, but usually they are:
    # 'Proximity', 'K_Feasibility(3)', 'DiscriminativePower(9)' (defaults)
    
    # Let's find the feasibility and discriminative power keys dynamically if possible, 
    # or just assume defaults for this example.
    feas_key = next((k for k in scores.keys() if "Feasibility" in k), "N/A")
    disc_key = next((k for k in scores.keys() if "DiscriminativePower" in k), "N/A")
    
    table.add_row(
        cf.explainer,
        str(cf.target_class),
        f"{scores.get('Proximity', 0):.4f}",
        f"{scores.get(feas_key, 0):.4f}" if feas_key != "N/A" else "N/A",
        f"{scores.get(disc_key, 0):.4f}" if disc_key != "N/A" else "N/A"
    )

console.print(table)
