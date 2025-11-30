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
        
        # We need to distinguish between original feature names and 
        # the names of the columns in the transformed (encoded) numpy array.
        self.original_feature_names = None
        self.transformed_feature_names = None
        
        self.neigh = None
        self.dataset_ref = None # Reference to dataset metadata

    def __repr__(self):
        return f"face_explainer(mode={self.mode}, fraction={self.fraction}, n_neighbors={self.n_neighbors})"

    def fit(self, model: Model, data: Dataset):
        """
        Fit FACE with a dataset and model.
        Builds a nearest-neighbor graph.
        """
        self.model = model
        self.dataset_ref = data
        
        self.X = np.array(data.data)
        self.y = np.array(data.target)
        self.original_feature_names = data.features

        # Reconstruct Transformed Feature Names
        self.transformed_feature_names = []
        
        self.transformed_feature_names.extend(data.continuous_features)
        
        for feat in data.categorical_features:
            categories = data.categorical_values.get(feat, [])
            encoded_names = [f"{feat}_{str(val)}" for val in categories]
            self.transformed_feature_names.extend(encoded_names)

        # Validation: Ensure our name generation matches the array width
        if len(self.transformed_feature_names) != self.X.shape[1]:
            print(f"[WARN] Feature name mismatch. Generated {len(self.transformed_feature_names)} names "
                  f"but data has {self.X.shape[1]} columns. Falling back to generic indices.")
            self.transformed_feature_names = [str(i) for i in range(self.X.shape[1])]

        print(self.transformed_feature_names)

        # 3. Subsampling (Optimization)
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
        
        df = pd.DataFrame(data.data, columns=self.transformed_feature_names)
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
        instance_df = pd.DataFrame(instance_ds.data, columns=self.transformed_feature_names)
        target = target_class or self.desired_class or 1
        return self._generate_cf(instance_df, model, target)

    def _generate_cf(self, instance_df: pd.DataFrame, model: Model, target_class: int) -> Optional[Counterfactual]:
        """
        Helper to generate a single counterfactual using the FACE approach.
        """
        try:
            # Extract the numpy array for the instance (1, n_features_transformed)
            instance = instance_df[self.transformed_feature_names].values[0]

            # Check Original Prediction
            pred_probs = model.predict_proba(instance.reshape(1, -1))
            pred_orig = np.argmax(pred_probs)
            
            if pred_orig == target_class:
                return None  # Already classified as desired

            # kNN search for similar points
            distances, indices = self.neigh.kneighbors(instance.reshape(1, -1))
            candidates = self.X[indices[0]]
            
            # Filter Candidates by Target Class
            candidate_probs = model.predict_proba(candidates)
            candidate_preds = np.argmax(candidate_probs, axis=1)
            
            valid_mask = (candidate_preds == target_class)
            valid_candidates = candidates[valid_mask]
            valid_distances = distances[0][valid_mask]

            if len(valid_candidates) == 0:
                return None

            # Select the Closest Valid Candidate (Shortest Path in Graph)
            best_idx = np.argmin(valid_distances)
            candidate_cf = valid_candidates[best_idx]

            return Counterfactual(
                original_data=instance,
                data=candidate_cf,
                original_class=pred_orig,
                target_class=target_class,
                explainer=repr(self),
            )

        except Exception as e:
            print(f"[WARN] FACE failed for instance: {e}")
            return None