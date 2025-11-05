import numpy as np
from typing import List, Optional
from tqdm import tqdm

from explainers_lib.counterfactual import Counterfactual
from explainers_lib.datasets import Dataset
from explainers_lib.explainers import Explainer
from explainers_lib.model import Model


class GrowingSpheresExplainer(Explainer):
    """
    Growing Spheres counterfactual explainer (Laugel et al., 2018),
    adapted for the library.
    """

    def __init__(self, step_size=0.2, max_iter=1000, num_samples=1000):
        self.step_size = step_size
        self.max_iter = max_iter
        self.num_samples = num_samples

    def __repr__(self) -> str:
        return f"growing_spheres(step_size={repr(self.step_size)}, max_iter={repr(self.max_iter)}, num_samples={repr(self.num_samples)})"

    def fit(self, model: Model, data: Dataset) -> None:
        # No fitting needed for Growing Spheres
        pass

    def _generate_counterfactual(
        self,
        instance_ds: Dataset,
        model: Model,
        target_class: int,
    ) -> Optional[Counterfactual]:
        radius = self.step_size
        instance = instance_ds.data[0]
        original_class = model.predict(instance_ds)[0]

        all_indices = list(range(len(instance_ds.features)))
        immutable_indices = instance_ds.immutable_features_ids
        continuous_indices = instance_ds.continuous_features_ids
        categorical_indices = instance_ds.categorical_features_ids
        
        mutable_indices = [i for i in all_indices if i not in immutable_indices]
        mutable_continuous_indices = [i for i in mutable_indices if i in continuous_indices]
        mutable_categorical_indices = [i for i in mutable_indices if i in categorical_indices]

        for i in range(self.max_iter):
            candidates = np.tile(instance, (self.num_samples, 1))
            
            # Handle continuous features
            if len(mutable_continuous_indices) > 0:
                directions = np.random.normal(size=(self.num_samples, len(mutable_continuous_indices)))
                directions /= np.linalg.norm(directions, axis=1, keepdims=True)
                
                continuous_updates = directions * radius
                candidates[:, mutable_continuous_indices] = instance[mutable_continuous_indices] + continuous_updates

            # Handle categorical features (assuming binary)
            if len(mutable_categorical_indices) > 0:
                categorical_updates = np.random.binomial(n=1, p=0.5, size=(self.num_samples, len(mutable_categorical_indices)))
                candidates[:, mutable_categorical_indices] = categorical_updates

            candidates_ds = instance_ds.like(candidates)

            pred_classes = model.predict(candidates_ds)
            
            valid_cf_indices = np.where(pred_classes == target_class)[0]

            if len(valid_cf_indices) > 0:
                distances = np.linalg.norm(candidates[valid_cf_indices] - instance, axis=1)
                best_cf_index = valid_cf_indices[np.argmin(distances)]
                
                return Counterfactual(
                    original_data=instance,
                    data=candidates[best_cf_index],
                    original_class=original_class,
                    target_class=target_class,
                    explainer=repr(self)
                )

            radius += self.step_size

        return None

    def explain_instance(
        self, instance_ds: Dataset, model: Model, target_class: Optional[int] = None
    ) -> Optional[Counterfactual]:
        original_class = model.predict(instance_ds)[0]
        
        if target_class is not None:
            if target_class == original_class:
                return None
            return self._generate_counterfactual(instance_ds, model, target_class)
        else:
            num_classes = getattr(instance_ds, "num_classes", 2)
            for c in range(num_classes):
                if c == original_class:
                    continue
                cf = self._generate_counterfactual(instance_ds, model, c)
                if cf is not None:
                    return cf
            return None

    def explain(self, model: Model, data: Dataset, y_desired: Optional[int] = None) -> List[Counterfactual]:
        counterfactuals: list[Counterfactual] = []

        for instance in tqdm(data, unit="instance"):
            cf = self.explain_instance(instance, model, y_desired)
            if cf is not None:
                counterfactuals.append(cf)
        
        return counterfactuals
