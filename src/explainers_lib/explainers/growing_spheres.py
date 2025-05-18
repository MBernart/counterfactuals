
from typing import Sequence

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

    def fit(self, model: Model, data: Dataset) -> None:
        # No fitting needed for Growing Spheres
        pass

    def explain(self, model: Model, data: Dataset) -> Sequence[Counterfactual]:
        counterfactuals = []

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

        while radius <= self.max_radius:
            directions = np.random.random((self.num_samples, dim))
            directions = directions / np.linalg.norm(directions) # unlikely for a random vector to have no length
            candidates = instance + directions * radius

            candidates_ds = instance_ds.like(candidates)

            # Get predictions for all candidates
            pred_classes = model.predict(candidates_ds)

            for i, pred_class in enumerate(pred_classes):
                if pred_class == target_class:
                    return Counterfactual(
                        original_data=instance,
                        changed_data=candidates[i],
                        original_class=original_class,
                        target_class=target_class,
                    )

            radius += self.step_size

        raise ValueError("No counterfactual found within max radius.")