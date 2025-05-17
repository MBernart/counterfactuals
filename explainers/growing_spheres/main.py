import json
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from explainers_lib.counterfactual import Counterfactual
from explainers_lib.explainers import Explainer
from explainers_lib.ensemble import Ensemble
from explainers_lib.datasets import Dataset
from explainers_lib.model import Model
import numpy as np
import torch
from pydantic import BaseModel
from typing import List, Sequence
from torch import nn


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
            # Convert the instance to a PyTorch Tensor
            instance_tensor = torch.tensor(instance, dtype=torch.float32)

            # Move the tensor to the same device as the model (if using CUDA)
            instance_tensor = instance_tensor.to(
                "cuda" if next(model.parameters()).is_cuda else "cpu"
            )  # model.device should be 'cpu' or 'cuda'

            # Get the predicted class by passing the instance through the model
            with torch.no_grad():  # Don't compute gradients during inference
                preds = model(instance_tensor.unsqueeze(0))  # Add batch dimension
                original_class = int(torch.argmax(preds))

            # Try to find a counterfactual for a different class
            for target_class in range(model(torch.rand(data[0].shape)).data.shape[0]):
                if target_class == original_class:
                    continue

                try:
                    cf = self._generate_counterfactual(
                        instance_tensor, model, target_class, original_class
                    )
                    counterfactuals.append(cf)
                    break  # Stop after finding the first valid CF
                except ValueError:
                    continue  # Try next target class

        return counterfactuals

    def _generate_counterfactual(
        self,
        instance: torch.Tensor,
        model: Model,
        target_class: int,
        original_class: int,
    ) -> Counterfactual:
        radius = self.step_size
        dim = instance.shape[0]

        while radius <= self.max_radius:
            directions = torch.randn(self.num_samples, dim).to(
                instance.device
            )  # Ensure same device
            directions = directions / directions.norm(
                dim=1, keepdim=True
            )  # Normalize directions
            candidates = instance + directions * radius

            # Get predictions for all candidates
            preds = model(candidates)  # Model should accept (num_samples, dim)
            pred_classes = torch.argmax(preds, dim=1)

            for i, pred_class in enumerate(pred_classes):
                if pred_class == target_class:
                    return Counterfactual(
                        original_data=instance.cpu().numpy(),
                        changed_data=candidates[i].cpu().numpy(),  # Convert tensor back to numpy array
                        original_class=original_class,
                        target_class=target_class,
                    )

            radius += self.step_size

        raise ValueError("No counterfactual found within max radius.")


class CounterfactualModel(BaseModel):
    """This is a helper class"""

    data: Sequence[float]
    original_class: int
    target_class: int


class InstanceRequest(BaseModel):
    data: Sequence[float]
    target_class: int


app = FastAPI()


@app.post("/growing-spheres/")
async def generate_growing_spheres_cf(
    instance_json: str = Form(...), model_file: UploadFile = File(...)
):
    instance_dict = json.loads(instance_json)
    instance = InstanceRequest(**instance_dict)

    model = torch.jit.load(model_file.file)

    gse = GrowingSpheresExplainer()
    gse.fit(model, None)
    instance_np = np.array(
        [
            instance.data,
        ]
    )
    original_class = instance.target_class

    cf = gse.explain(model, instance_np)

    target_class = model(torch.tensor(cf[0].changed_data)).argmax().item()
    print(model(torch.tensor(cf[0].changed_data)))

    return CounterfactualModel(
        data=cf[0].changed_data.tolist(),
        original_class=original_class,
        target_class=target_class,
    )
