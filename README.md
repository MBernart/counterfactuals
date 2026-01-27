# A Framework for Counterfactual Explanations

This repository contains the source code for a framework for generating and evaluating counterfactual explanations (CFEs). The framework is designed to be scalable and extensible, allowing for the integration of new CFE methods and evaluation metrics. It also provides tools for ensembling multiple CFE methods to generate more robust and diverse explanations.

## Project Structure

The project is organized as follows:

- `src/explainers_lib`: The core library code.
- `examples`: Example usage of the library.
- `experiments`: Jupyter notebooks for experiments and analysis.
- `explainers`: Dockerized explainers.

## Installation

### For local development

```shell
pip install -e ./src
```

### Using Docker for explainers

Some explainers can be run as dockerized workers. For example, to run the Wachter explainer:

1. Start Redis message broker:
   ```shell
   docker run -d -p 6379:6379 --name celery-redis redis
   ```
2. Pull the explainer image or build it:

   ```shell
   # Pull from registry
   docker pull cfe.cs.put.poznan.pl/counterfactuals-wachter

   # Or build it yourself
   docker build -t wachter-explainer -f explainers/native/Dockerfile .
   ```

3. Run the explainer:
   ```shell
   docker run --rm -it --network host wachter-explainer
   ```

## Usage

Here is a basic example of how to use the library to generate counterfactual explanations:

```python
from explainers_lib.explainers.native.wachter import WachterExplainer
from explainers_lib import TorchModel, Dataset, postprocess_cfs, print_cfs
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

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
with open("examples/temp_model.pt", "rb") as f:
    model_data = f.read()
model = TorchModel.deserialize(model_data)

# Running the explainer
explainer = WachterExplainer()
explainer.fit(model, data)
cfs = explainer.explain(model, data[:5])

# Data postprocessing
cfs = postprocess_cfs(cfs, data.inverse_transform, label_encoder.inverse_transform)

print_cfs(cfs, feature_names=data.features)
```

## Ensemble Usage

You can also use an ensemble of explainers to generate counterfactual explanations from multiple sources.

```python
from explainers_lib.explainers.native.wachter import WachterExplainer
from explainers_lib.explainers.native.growing_spheres import GrowingSpheresExplainer
from explainers_lib.explainers.celery_explainer import ActionableRecourseExplainer
from explainers_lib.explainers.celery_explainer import FaceExplainer
from explainers_lib.explainers.celery_explainer import DiceExplainer
from explainers_lib.explainers.celery_explainer import AlibiCFProto
from explainers_lib.explainers.celery_explainer import AlibiCFRL
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
with open("examples/temp_model.pt", "rb") as f:
    model_data = f.read()

model = TorchModel.deserialize(model_data)

# Ensemble
ensemble = Ensemble(
    model,
    [
        # Native
        WachterExplainer(),
        GrowingSpheresExplainer(),
        FaceExplainer(),
        # Dice
        DiceExplainer(),
        # Alibi
        AlibiCFProto(),
        AlibiCFRL()
    ],
    Pareto())
print(f"Used celery explainers: {[explainer.explainer_name for explainer in ensemble.celery_explainers]}")

ensemble.fit(data)
print(f"Ensemble fitting complete")

cfs = ensemble.explain(data[:5],
                       pretty_print=True,
                       pretty_print_postprocess=data.inverse_transform,
                       pretty_print_postprocess_target=label_encoder.inverse_transform)
print(f"Number of generated cfs: {len(cfs)}")
```

## License

This project is licensed under the MIT License.
