from typing import Sequence
from .counterfactual import Counterfactual
from .datasets import Dataset, SerializableDataset
from .model import Model, SerializableModel
import numpy as np
import torch


class Explainer:
    """This is an abstract class for an explainer"""

    def fit(self, model: Model, data: Dataset) -> None:
        """This method is used to fit the explainer"""
        raise NotImplementedError

    # probably we want to explain (find counterfactuals for) a specific record (can be extended to subset)
    def explain(self, model: Model, data: Dataset, record: int) -> Sequence[Counterfactual]:
        """This method is used generate the counterfactuals"""
        raise NotImplementedError


class RemoteExplainer:
    def connect(ip: str) -> None:
        raise NotImplementedError

    def fit(self, model: SerializableModel, data: SerializableDataset) -> None:
        """This method is used to fit the explainer"""
        raise NotImplementedError

    def explain(
        self, model: SerializableModel, data: SerializableDataset, record: int
    ) -> Sequence[Counterfactual]:
        """This method is used generate the counterfactuals"""
        raise NotImplementedError


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
                        data=candidates[i]
                        .cpu()
                        .numpy(),  # Convert tensor back to numpy array
                        original_class=original_class,
                        target_class=target_class,
                    )

            radius += self.step_size

        raise ValueError("No counterfactual found within max radius.")
