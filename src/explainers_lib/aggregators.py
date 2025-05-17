from abc import ABC, abstractmethod  # proposed by gpt
from typing import Callable, Sequence, TypeAlias
from .counterfactual import Counterfactual
from utils.scores import get_scores
from utils.pareto import get_pareto_optimal_mask

# Aggregator: TypeAlias = Callable[[Sequence[Counterfactual]], Counterfactual]


# Type alias (for convenience or registration)
Aggregator = Callable[[Sequence[Counterfactual]], Sequence[Counterfactual]]


class AggregatorBase(ABC):
    """Abstract base class for counterfactual aggregators"""

    @abstractmethod
    def __call__(self, cfs: Sequence[Counterfactual]) -> Sequence[Counterfactual]:
        pass

# run experiment.py
class IdealPoint(AggregatorBase):
    """Computes the ideal point from counterfactuals"""

    def __call__(self, cfs: Sequence[Counterfactual]) -> Sequence[Counterfactual]:
        # Example: one "ideal" counterfactual minimizing changes
        
        raise NotImplementedError


class Pareto(AggregatorBase):
    """Computes the pareto front from counterfactuals"""

    def __call__(self, cfs: Sequence[Counterfactual]) -> Sequence[Counterfactual]:
        # Example: return all Pareto-efficient counterfactuals
        raise NotImplementedError
    

# my idea
class All(AggregatorBase):
    """Return all (valid) counterfactuals found by explainer"""

    def __call__(self, cfs: Sequence[Counterfactual]) -> Sequence[Counterfactual]:
        # Example: return all counterfactuals
        return cfs
