import json
from explainers_lib.counterfactual import ClassLabel, Counterfactual
from explainers_lib.explainers import Explainer, WorkerFactory
from explainers_lib.ensemble import Ensemble
from explainers_lib.datasets import Dataset
from explainers_lib.model import Model, SerializableModel
import numpy as np
from typing import List, Sequence


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
        for instance in data:
            instance_ds = Dataset(np.array([instance]), [0], data.features, data.immutable_features, data.categorical_features, data.allowable_ranges)

            original_class = model.predict(instance_ds)[0]

            # Try to find a counterfactual for a different class
            for target_class in range(len(set(data.target))):
                if target_class == original_class:
                    continue

                try:
                    cf = self._generate_counterfactual(instance_ds, model, target_class, original_class)
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
        instance = next(instance for instance in instance_ds)
        dim = instance.shape[0]

        while radius <= self.max_radius:
            directions = np.random.random((self.num_samples, dim))
            directions = directions / np.linalg.norm(directions) # unlikely for a random vector to have no length
            candidates = instance + directions * radius

            candidates_ds = Dataset(candidates, [], instance_ds.features, instance_ds.immutable_features, instance_ds.categorical_features, instance_ds.allowable_ranges)

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

from twisted.internet import reactor, protocol, defer, task
from twisted.protocols.basic import LineReceiver
import pickle
import sys
import time

reactor.listenTCP(8000, WorkerFactory())

reactor.run()

# app = FastAPI()


# @app.post("/growing-spheres/")
# async def generate_growing_spheres_cf(
#     instance_json: str = Form(...), model_file: UploadFile = File(...)
# ):
#     instance_dict = json.loads(instance_json)
#     instance = InstanceRequest(**instance_dict)

#     model = torch.jit.load(model_file.file)

#     gse = GrowingSpheresExplainer()
#     gse.fit(model, None)
#     instance_np = np.array(
#         [
#             instance.data,
#         ]
#     )
#     original_class = instance.target_class

#     cf = gse.explain(model, instance_np)

#     target_class = model(torch.tensor(cf[0].changed_data)).argmax().item()
#     print(model(torch.tensor(cf[0].changed_data)))

#     return CounterfactualModel(
#         data=cf[0].changed_data.tolist(),
#         original_class=original_class,
#         target_class=target_class,
#     )
