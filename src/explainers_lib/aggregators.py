import pandas as pd
import numpy as np
from abc import ABC, abstractmethod  # proposed by gpt
from typing import Callable, TypeAlias
from .counterfactual import Counterfactual
from .utils.scores import get_scores
from .utils.pareto import get_pareto_optimal_mask


# Aggregator: TypeAlias = Callable[[Sequence[Counterfactual]], Counterfactual]


# Type alias (for convenience or registration)
Aggregator = Callable[[list[Counterfactual]], list[Counterfactual]]


class AggregatorBase(ABC):
    """Abstract base class for counterfactual aggregators"""

    @abstractmethod
    def __call__(self, cfs: list[Counterfactual]) -> list[Counterfactual]:
        pass



# run experiment.py
class IdealPoint(AggregatorBase):
    """Computes the ideal point from counterfactuals"""

    def __call__(self, cfs: list[Counterfactual]) -> list[Counterfactual]:
        # Example: one "ideal" counterfactual minimizing changes
        
        raise NotImplementedError


class Pareto(AggregatorBase):
    """Computes the pareto front from counterfactuals"""

    def __call__(self, cfs: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
        # Example: return all Pareto-efficient counterfactuals
        x_metric = 'Proximity'
        y_metric = 'K_Feasibility(3)'
        z_metric = 'DiscriminativePower(9)'
        optimization_directions = ['min', 'min', 'max']

        all_x = scores[x_metric].to_numpy()
        all_y = scores[y_metric].to_numpy()
        all_z = scores[z_metric].to_numpy()
        to_check = np.array([all_x, all_y, all_z], dtype=np.float64).T
        pareto_mask = get_pareto_optimal_mask(data=to_check, optimization_direction=optimization_directions).astype('bool')

        pareto_front = np.array(cfs)[pareto_mask.astype(bool)]
        return pareto_front

        # raise NotImplementedError
    

# my idea
class All(AggregatorBase):
    """Return all (valid) counterfactuals found by explainer"""

    def __call__(self, cfs: list[Counterfactual]) -> list[Counterfactual]:
        # Example: return all counterfactuals
        return cfs
