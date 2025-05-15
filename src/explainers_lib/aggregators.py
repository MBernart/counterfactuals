from abc import ABC, abstractmethod  # proposed by gpt
from typing import Callable, TypeAlias
from .counterfactual import Counterfactual


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

    def __call__(self, cfs: list[Counterfactual]) -> list[Counterfactual]:
        # Example: return all Pareto-efficient counterfactuals
        raise NotImplementedError
    

# my idea
class All(AggregatorBase):
    """Return all (valid) counterfactuals found by explainer"""

    def __call__(self, cfs: list[Counterfactual]) -> list[Counterfactual]:
        # Example: return all counterfactuals
        return cfs
