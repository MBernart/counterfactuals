import pandas as pd
import numpy as np
from abc import ABC, abstractmethod  # proposed by gpt
from typing import Callable, List
from .utils.scores import get_scores
from .model import Model
from .datasets import Dataset
from .counterfactual import Counterfactual
from .utils.pareto import get_pareto_optimal_mask, get_ideal_point


# Aggregator: TypeAlias = Callable[[Sequence[Counterfactual]], Counterfactual]


# Type alias (for convenience or registration)
Aggregator = Callable[[List[Counterfactual]], List[Counterfactual]]


class AggregatorBase(ABC):
    """Abstract base class for counterfactual aggregators"""

    @abstractmethod
    def __call__(self, cfs: List[Counterfactual]) -> List[Counterfactual]:
        pass


class Pareto(AggregatorBase):
    """Computes the Pareto front from counterfactuals"""

    def __init__(self, k_neigh_feasibility = 3, k_neigh_discriminative = 9):
        self.k_neigh_feasibility = k_neigh_feasibility
        self.k_neigh_discriminative = k_neigh_discriminative

    def fit(self, model: Model, data: Dataset) -> None:
        self.model = model
        self.data = data
        self.train_preds = self.model.predict(self.data)

    def calculate_scores(self, cfs: List[Counterfactual]) -> pd.DataFrame:
        original_data = cfs[0].original_data.reshape(1, -1)
        cfs_data = np.array([cf.data for cf in cfs])
        cfs_target = np.array([cf.target_class for cf in cfs])

        return get_scores(
            cfs=cfs_data,
            cf_predicted_classes=cfs_target,
            training_data=self.data.data,
            training_data_predicted_classes=self.train_preds,
            x = original_data,
            continous_indices=self.data.continuous_features_ids,
            categorical_indices=self.data.categorical_features_ids,
            k_neighbors_feasib=self.k_neigh_feasibility, 
            k_neighbors_discriminative=self.k_neigh_discriminative
            ).reset_index(drop=True)

    def __call__(self, cfs: List[Counterfactual]) -> List[Counterfactual]:
        scores = self.calculate_scores(cfs)

        # Example: return all Pareto-efficient counterfactuals
        x_metric = 'Proximity'
        y_metric = 'K_Feasibility(3)'
        z_metric = 'DiscriminativePower(9)'
        optimization_direction = ['min', 'min', 'max']

        all_x = scores[x_metric].to_numpy()
        all_y = scores[y_metric].to_numpy()
        all_z = scores[z_metric].to_numpy()
        to_check = np.array([all_x, all_y, all_z], dtype=np.float64).T

        pareto_mask = get_pareto_optimal_mask(
            data=to_check,
            optimization_direction=optimization_direction
        ).astype(bool)

        pareto_indices = np.where(pareto_mask)[0]
        pareto_cfs = [cfs[i] for i in pareto_indices]
        return pareto_cfs


class IdealPoint(Pareto):
    """Computes the ideal point from counterfactuals"""
    def __init__(self, weights: List[float] = None, k_neigh_feasibility = 3, k_neigh_discriminative = 9):
        """
        weights: optional list of 3 weights for x, y, z metrics
        (will be normalized to sum = 1). If None, equal weights are used.
        """
        super().__init__(k_neigh_feasibility=k_neigh_feasibility, k_neigh_discriminative=k_neigh_discriminative)
        self.weights = weights

    def __call__(self, cfs: List[Counterfactual]) -> List[Counterfactual]:
        scores = self.calculate_scores(cfs)

        x_metric = 'Proximity'
        y_metric = 'K_Feasibility(3)'
        z_metric = 'DiscriminativePower(9)'
        optimization_direction = ['min', 'min', 'max']

        all_x = scores[x_metric].to_numpy()
        all_y = scores[y_metric].to_numpy()
        all_z = scores[z_metric].to_numpy()
        to_check = np.array([all_x, all_y, all_z], dtype=np.float64).T

        pareto_mask = get_pareto_optimal_mask(
            data=to_check,
            optimization_direction=optimization_direction
        ).astype(bool)
        
        pareto_indices = np.where(pareto_mask)[0]
        pareto_cfs = [cfs[i] for i in pareto_indices]
        pareto_data = to_check[pareto_mask]

        ideal_point = get_ideal_point(to_check, optimization_direction, pareto_mask)

        if self.weights is None:
            weights = np.ones(to_check.shape[1]) / to_check.shape[1]
        else:
            weights = np.array(self.weights, dtype=float)
            weights = weights / weights.sum()  # normalize

        # weighted distances
        diffs = pareto_data - ideal_point
        dists = np.sqrt(np.sum(weights * diffs**2, axis=1))
        # pick closest
        best_idx = np.argmin(dists)
        return [pareto_cfs[best_idx]]


class All(AggregatorBase):
    """Return all (valid) counterfactuals found by explainer"""

    def __call__(self, cfs: List[Counterfactual]) -> List[Counterfactual]:
        return cfs
