import numpy as np
from tqdm import tqdm
from explainers_lib.counterfactual import Counterfactual
from explainers_lib.datasets import Dataset
from explainers_lib.explainers import Explainer
from explainers_lib.model import Model


class GrowingSpheresExplainer(Explainer):
    def __init__(self, step_size=0.1, max_radius=5.0, num_samples=1000):
        self.step_size = step_size
        self.max_radius = max_radius
        self.num_samples = num_samples

    def __repr__(self) -> str:
        return f"growing_spheres(step_size={repr(self.step_size)}, max_radius={repr(self.max_radius)}, num_samples={repr(self.num_samples)})"

    def fit(self, model: Model, data: Dataset) -> None:
        # No fitting needed for Growing Spheres
        pass

    def explain(self, model: Model, data: Dataset) -> list[Counterfactual]:
        counterfactuals: list[Counterfactual] = []

        # Assuming data is an iterable, for each instance
        for instance in tqdm(data, unit="instance"):

            original_class = model.predict(instance)[0]

            # Try to find a counterfactual for a different class
            for target_class in range(len(set(data.target))):
                if target_class == original_class:
                    continue

                try:
                    cf = self._generate_counterfactual(instance, model, target_class, original_class)
                    counterfactuals.append(cf)
                    break  # Stop after finding the first valid CF
                except ValueError:
                    continue  # Try next target class

        return counterfactuals

    def _generate_counterfactual(
        self,
        instance_ds: Dataset,
        model: Model,
        target_class: int,
        original_class: int,
    ) -> Counterfactual:
        radius = self.step_size
        instance = instance_ds.data[0]
        dim = instance.shape[0]

        immutable_transformed_indices = []

        continuous_immutable_indices = [
            i
            for i, f in enumerate(instance_ds.continuous_features)
            if f in instance_ds.immutable_features
        ]
        immutable_transformed_indices.extend(continuous_immutable_indices)

        if instance_ds.categorical_features:
            ohe = instance_ds.preprocessor.named_transformers_["cat"].named_steps[
                "onehot"
            ]
            cat_feature_names = instance_ds.categorical_features
            cat_immutable_features = [
                f for f in instance_ds.immutable_features if f in cat_feature_names
            ]

            if cat_immutable_features:
                offset = len(instance_ds.continuous_features)
                n_cats_per_feature = [len(cats) for cats in ohe.categories_]

                cat_indices_start = np.cumsum([0] + n_cats_per_feature[:-1])

                for f in cat_immutable_features:
                    idx_in_cat_list = cat_feature_names.index(f)
                    start = offset + cat_indices_start[idx_in_cat_list]
                    end = start + n_cats_per_feature[idx_in_cat_list]
                    immutable_transformed_indices.extend(range(start, end))

        while radius <= self.max_radius:
            directions = np.random.random((self.num_samples, dim))
            norm = np.linalg.norm(directions, axis=1, keepdims=True)
            norm[norm == 0] = 1e-9
            directions = directions / norm
            candidates = instance + directions * radius

            if immutable_transformed_indices:
                candidates[:, immutable_transformed_indices] = instance[
                    immutable_transformed_indices
                ]

            if instance_ds.allowable_ranges:
                df_candidates = instance_ds.inverse_transform(candidates)
                for feature, (min_val, max_val) in instance_ds.allowable_ranges.items():
                    if feature in df_candidates.columns:
                        df_candidates[feature] = df_candidates[feature].clip(
                            min_val, max_val
                        )
                candidates = instance_ds.preprocessor.transform(df_candidates)

            # Get predictions for all candidates
            pred_classes = model.predict(candidates)

            for i, pred_class in enumerate(pred_classes):
                if pred_class == target_class:
                    return Counterfactual(instance, candidates[i], original_class, pred_class, repr(self))

            radius += self.step_size

        raise ValueError("No counterfactual found within max radius.")