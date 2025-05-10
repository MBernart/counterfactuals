from typing import Callable, Sequence, TypeAlias
from .counterfactual import Counterfactual

Aggregator: TypeAlias = Callable[[Sequence[Counterfactual]], Counterfactual]


class IdealPoint(Aggregator):
    """Computes the ideal point from counterfactuals"""

    def __call__(self, cfs: Sequence[Counterfactual]) -> Sequence[Counterfactual]:
        raise NotImplementedError


class Pareto(Aggregator):
    """Computes the pareto front from counterfactuals"""

    def __call__(self, cfs: Sequence[Counterfactual]) -> Sequence[Counterfactual]:
        raise NotImplementedError
