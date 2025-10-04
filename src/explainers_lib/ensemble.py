from typing import Any
from celery import chain, group
from .model import Model
from .explainers import Explainer
from .explainers.celery_remote import app, try_get_available_explainers
from .explainers.celery_explainer import CeleryExplainer
from .aggregators import Aggregator, All, Pareto
from .counterfactual import Counterfactual
from .datasets import Dataset


class Ensemble:
    def __init__(
        self, model: Model, explainers: list[Explainer], aggregator: Aggregator = All()
    ):
        """Constructs an ensemble"""
        self.model = model
        self.aggregator = aggregator

        self.explainers = list(filter(lambda explainer: not isinstance(explainer, CeleryExplainer), explainers))
        self.celery_explainers = list(filter(lambda explainer: isinstance(explainer, CeleryExplainer), explainers))
        self.celery_explainers = ensure_celery_explainers(self.celery_explainers)

    def fit(self, data: Dataset) -> None:
        """This method is used to train all explainers in the ensemble"""
        
        task = fit_celery_explainers(self.celery_explainers, self.model, data)

        for explainer in self.explainers:
            explainer.fit(self.model, data)

        if isinstance(self.aggregator, Pareto):
            self.aggregator.fit(self.model, data)

        if task:
            task.get()

    def explain(self, data: Dataset) -> list[Counterfactual]:
        """This method is used to generate counterfactuals"""

        task = explain_celery_explainers(self.celery_explainers, self.model, data)

        all_counterfactuals = list()
        for explainer in self.explainers:
            cfs = explainer.explain(self.model, data)
            all_counterfactuals.extend(cfs)

        if task:
            results = task.get()
            cfs = [Counterfactual.deserialize(counterfactual) for result in results for counterfactual in result['counterfactuals']]
            all_counterfactuals.extend(cfs)

        if isinstance(self.aggregator, Pareto):
            self.aggregator.query_instance = data # TODO: this is a hack
        
        return self.aggregator(all_counterfactuals)

def ensure_celery_explainers(requested_explainers: list[CeleryExplainer]) -> list[CeleryExplainer]:
    if len(requested_explainers) == 0:
        return []

    available_explainers = try_get_available_explainers()
    explainers_set = set(explainer.explainer_name for explainer in requested_explainers)

    explainers = list(filter(lambda explainer: explainer.explainer_name in available_explainers, requested_explainers))
    missing_explainers = list(explainers_set - available_explainers)

    if len(explainers) == 0:
        raise RuntimeError(f"Explainers not found: {missing_explainers}")
    
    if len(missing_explainers) > 0:
        raise RuntimeWarning(f"Explainers not found: {missing_explainers}")
    
    return explainers

def fit_celery_explainers(explainers: list[CeleryExplainer], model: Model, data: Dataset) -> Any | None:
    if len(explainers) == 0:
        return None

    fit_chains = []
    for explainer in explainers:
        fit_chains.append(explainer.fit_async(model, data))

    return chain([group(fit_chains), app.signature('ensemble.collect_results')]).apply_async()

def explain_celery_explainers(explainers: list[CeleryExplainer], model: Model, data: Dataset) -> Any | None:
    if len(explainers) == 0:
        return None

    explain_chains = []
    for explainer in explainers:
        explain_chains.append(explainer.explain_async(model, data))

    return chain([group(explain_chains) | app.signature('ensemble.collect_results')]).apply_async()
