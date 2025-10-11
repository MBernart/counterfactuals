import numpy as np
import pandas as pd
from typing import List, Optional
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from explainers_lib.datasets import Dataset
from explainers_lib.explainers import Explainer
from explainers_lib.counterfactual import Counterfactual
from explainers_lib.model import Model


class FaceExplainer(Explainer):
    """
    FACE-based counterfactual explainer (Poyiadzi et al., 2020),
    implemented without CARLA dependencies.
    """

    def __init__(
        self,
        mode: str = "knn",
        fraction: float = 0.05,
        desired_class: Optional[int] = None,
        n_neighbors: int = 50,
    ):
        """
        Args:
            mode: Graph-building mode ("knn" or "epsilon").
            fraction: Fraction of data to construct neighborhood graph.
            desired_class: Desired output class for generated counterfactuals.
            n_neighbors: Number of neighbors for kNN search.
        """
        self.mode = mode
        self.fraction = fraction
        self.desired_class = desired_class
        self.n_neighbors = n_neighbors
        self.model = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.neigh = None

    def __repr__(self):
        return f"face_explainer(mode={self.mode}, fraction={self.fraction}, n_neighbors={self.n_neighbors})"

    def fit(self, model: Model, data: Dataset):
        """
        Fit FACE with a dataset and model.
        Builds a nearest-neighbor graph.
        """
        self.model = model
        self.X = np.array(data.data)
        self.y = np.array(data.target)
        self.feature_names = data.features

        # Optionally subsample dataset to fraction
        if 0 < self.fraction < 1:
            n = max(2, int(len(self.X) * self.fraction))  # at least 2 samples
            if n < len(self.X):
                idx = np.random.choice(len(self.X), size=n, replace=False)
                self.X = self.X[idx]
                self.y = self.y[idx]

        # Adjust n_neighbors so it’s never greater than number of samples
        effective_neighbors = min(self.n_neighbors, len(self.X))

        # Build kNN structure
        self.neigh = NearestNeighbors(n_neighbors=effective_neighbors)
        self.neigh.fit(self.X)


    def explain(self, model: Model, data: Dataset, y_desired: Optional[int] = None) -> List[Counterfactual]:
        """
        Generate counterfactuals for multiple instances.
        """
        counterfactuals = []
        df = pd.DataFrame(data.data, columns=data.features)
        y_target = y_desired or self.desired_class or 1

        for i in tqdm(range(len(df)), unit="instance"):
            instance_df = df.iloc[[i]]
            cf = self._generate_cf(instance_df, model, y_target)
            if cf is not None:
                counterfactuals.append(cf)

        return counterfactuals

    def explain_instance(
        self, instance_ds: Dataset, model: Model, target_class: Optional[int] = None
    ) -> Optional[Counterfactual]:
        """
        Generate counterfactual for a single instance.
        """
        instance_df = pd.DataFrame(instance_ds.data, columns=self.feature_names)
        target = target_class or self.desired_class or 1
        return self._generate_cf(instance_df, model, target)

    def _generate_cf(self, instance_df: pd.DataFrame, model: Model, target_class: int) -> Optional[Counterfactual]:
        """
        Helper to generate a single counterfactual using the FACE approach.
        """
        try:
            instance = instance_df[self.feature_names].values[0]

            pred_orig = np.argmax(model.predict_proba(instance.reshape(1, -1)))
            if pred_orig == target_class:
                return None  # Already classified as desired

            # kNN search for similar points
            distances, indices = self.neigh.kneighbors(instance.reshape(1, -1))
            candidates = self.X[indices[0]]
            preds = np.argmax(model.predict_proba(candidates), axis=1)

            # Filter to those in desired class
            valid_idx = np.where(preds == target_class)[0]
            if len(valid_idx) == 0:
                return None

            # Select the closest valid candidate
            best_idx = valid_idx[np.argmin(distances[0][valid_idx])]
            candidate = candidates[best_idx]

            pred_cf = target_class
            return Counterfactual(
                original_data=instance,
                data=candidate,
                original_class=pred_orig,
                target_class=pred_cf,
                explainer=repr(self),
            )

        except Exception as e:
            print(f"[WARN] FACE failed for instance: {e}")
            return None