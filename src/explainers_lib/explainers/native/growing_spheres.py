import numpy as np
from tqdm import tqdm
from explainers_lib.counterfactual import Counterfactual
from explainers_lib.datasets import Dataset
from explainers_lib.explainers import Explainer
from explainers_lib.model import Model


class GrowingSpheresExplainer(Explainer):
    def __init__(self, step_size=0.1, max_radius=5.0, num_samples=1000, k=1):
        self.step_size = step_size
        self.max_radius = max_radius
        self.num_samples = num_samples
        self.k = k

    def __repr__(self) -> str:
        return f"growing_spheres(step_size={repr(self.step_size)}, max_radius={repr(self.max_radius)}, num_samples={repr(self.num_samples)}, k={repr(self.k)})"

    def fit(self, model: Model, data: Dataset) -> None:
        # No fitting needed for Growing Spheres
        pass

    def explain(
        self, model: Model, data: Dataset, y_desired: int = None
    ) -> list[Counterfactual]:
        counterfactuals: list[Counterfactual] = []

        # Assuming data is an iterable, for each instance
        for instance in tqdm(data, unit="instance"):

            original_class = model.predict(instance)[0]

            # Try to find a counterfactual for a different class
            if y_desired == original_class:
                continue

            cfs = self._generate_counterfactual(
                instance, model, y_desired, original_class
            )
            counterfactuals.extend(cfs)

        return counterfactuals

    def _generate_counterfactual(
        self,
        instance_ds: Dataset,
        model: Model,
        target_class: int,
        original_class: int,
    ) -> list[Counterfactual]:
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

        cat_feature_slices = []
        if instance_ds.categorical_features:
            ohe = instance_ds.preprocessor.named_transformers_["cat"].named_steps[
                "onehot"
            ]
            n_cats_per_feature = [len(cats) for cats in ohe.categories_]
            offset = len(instance_ds.continuous_features)
            cat_indices_start = np.cumsum([0] + n_cats_per_feature[:-1])

            for i, start in enumerate(cat_indices_start):
                end = start + n_cats_per_feature[i]
                cat_feature_slices.append(slice(offset + start, offset + end))

            cat_feature_names = instance_ds.categorical_features
            cat_immutable_features = [
                f for f in instance_ds.immutable_features if f in cat_feature_names
            ]

            if cat_immutable_features:
                for f in cat_immutable_features:
                    idx_in_cat_list = cat_feature_names.index(f)
                    start = offset + cat_indices_start[idx_in_cat_list]
                    end = start + n_cats_per_feature[idx_in_cat_list]
                    immutable_transformed_indices.extend(range(start, end))

        result: list[Counterfactual] = []
        while radius <= self.max_radius and len(result) < self.k:
            directions = np.random.random((self.num_samples, dim))
            norm = np.linalg.norm(directions, axis=1, keepdims=True)
            norm[norm == 0] = 1e-9
            directions = directions / norm
            candidates = instance + directions * radius

            if cat_feature_slices:
                for s in cat_feature_slices:
                    cat_part = candidates[:, s]
                    max_indices = np.argmax(cat_part, axis=1)
                    new_cat_part = np.zeros_like(cat_part)
                    new_cat_part[np.arange(len(new_cat_part)), max_indices] = 1
                    candidates[:, s] = new_cat_part

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
            pred_classes = np.asarray(model.predict(candidates))

            mask = pred_classes != original_class
            if target_class is not None:
                mask &= (pred_classes == target_class)

            valid_indices = np.where(mask)[0]
            needed = self.k - len(result)

            result.extend([
                Counterfactual(
                    instance, candidates[i], original_class, pred_classes[i], repr(self)
                )
                for i in valid_indices[:needed]
            ])

            radius += self.step_size

        return result
