from celery import group
from .model import Model
from .explainers import Explainer
from .explainers.celery_remote import app, try_get_available_explainers
from .explainers.celery_explainer import CeleryExplainer
from .aggregators import Aggregator, All, Pareto
from .counterfactual import Counterfactual
from .datasets import Dataset


class Ensemble:
    def __init__(
        self, model: Model, explainers: list[Explainer], aggregator: Aggregator
    ):
        """Constructs an ensemble"""
        self.model = model
        self.explainers = explainers
        self.aggregator = aggregator

    def fit(self, data: Dataset) -> None:
        """This method is used to train all explainers in the ensemble"""
        for explainer in self.explainers:
            explainer.fit(self.model, data)

        if isinstance(self.aggregator, Pareto):
            self.aggregator.fit(self.model, data)

    def explain(self, data: Dataset) -> list[Counterfactual]:
        """This method is used to generate counterfactuals"""

        all_counterfactuals = list()
        for explainer in self.explainers:
            cfs = explainer.explain(self.model, data)
            all_counterfactuals.extend(cfs)

        if isinstance(self.aggregator, Pareto):
            self.aggregator.query_instance = data # TODO: this is a hack
        
        return self.aggregator(all_counterfactuals)

# TODO: merge CeleryEnsemble and Ensemble
class CeleryEnsemble:
    def __init__(
        self, model: Model, explainers: list[CeleryExplainer], aggregator: Aggregator = All()
    ):
        """Constructs an ensemble"""

        self.model = model
        self.aggregator = aggregator

        available_explainers = try_get_available_explainers()
        explainers_set = set(explainer.explainer_name for explainer in explainers)

        self.explainers = list(filter(lambda explainer: explainer.explainer_name in available_explainers, explainers))
        missing_explainers = list(explainers_set - available_explainers)

        if len(self.explainers) == 0:
            raise RuntimeError(f"CeleryEnsemble: Explainers not found: {missing_explainers}")
        
        if len(missing_explainers) > 0:
            raise RuntimeWarning(f"CeleryEnsemble: Explainers not found: {missing_explainers}")

    def fit(self, data: Dataset) -> None:
        """This method is used to train all explainers in the ensemble"""

        fit_chains = []
        for explainer in self.explainers:
            fit_chains.append(explainer.fit_async(self.model, data))

        fit_chord = group(fit_chains) | app.signature('ensemble.collect_results')
        fit_chord.apply_async().get()

        if isinstance(self.aggregator, Pareto):
            self.aggregator.fit(self.model, data)

    def explain(self, data: Dataset) -> list[Counterfactual]:
        """This method is used to generate counterfactuals"""

        explain_chains = []
        for explainer in self.explainers:
            explain_chains.append(explainer.explain_async(self.model, data))

        explain_chord = group(explain_chains) | app.signature('ensemble.collect_results')
        results = explain_chord.apply_async().get()

        counterfactuals = [Counterfactual.deserialize(counterfactual) for result in results for counterfactual in result['counterfactuals']]

        if isinstance(self.aggregator, Pareto):
            self.aggregator.query_instance = data # TODO: this is a hack

        return self.aggregator(counterfactuals)
