import pandas as pd
import numpy as np
from abc import ABC, abstractmethod  # proposed by gpt
from typing import Callable, TypeAlias
from .counterfactual import Counterfactual
from .utils.pareto import get_pareto_optimal_mask, get_ideal_point


# Aggregator: TypeAlias = Callable[[Sequence[Counterfactual]], Counterfactual]


# Type alias (for convenience or registration)
Aggregator = Callable[[list[Counterfactual]], list[Counterfactual]]


class AggregatorBase(ABC):
    """Abstract base class for counterfactual aggregators"""

    @abstractmethod
    def __call__(self, cfs: list[Counterfactual]) -> list[Counterfactual]:
        pass


class Pareto(AggregatorBase):
    """Computes the Pareto front from counterfactuals"""

    def __call__(self, cfs: pd.DataFrame, scores: pd.DataFrame):
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

        pareto_cfs = cfs[pareto_mask]
        return pareto_cfs


class IdealPoint(AggregatorBase):
    """Computes the ideal point from counterfactuals"""
    def __init__(self, weights: list[float] = None):
        """
        weights: optional list of 3 weights for x, y, z metrics
        (will be normalized to sum = 1). If None, equal weights are used.
        """
        self.weights = weights

    def __call__(self, cfs: pd.DataFrame, scores: pd.DataFrame):
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
            pareto_cfs = cfs[pareto_mask]
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
            return pareto_cfs.iloc[[best_idx]]



class All(AggregatorBase):
    """Return all (valid) counterfactuals found by explainer"""

    def __call__(self, cfs: pd.DataFrame, scores: pd.DataFrame):
        return cfs
