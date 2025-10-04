from celery import group
import pandas as pd
from .model import Model
from .explainers import Explainer
from .explainers.celery_remote import app, try_get_available_explainers
from .explainers.celery_explainer import CeleryExplainer
from .aggregators import Aggregator, All
from .counterfactual import Counterfactual
from .datasets import Dataset
from .utils.scores import get_scores


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

    # probably want to explain single record at once
    def explain(self, query_instance: Dataset, data: Dataset) -> pd.DataFrame:
        """This method is used to generate counterfactuals"""

        all_counterfactuals = pd.DataFrame(columns=data.features + ['target'])
        for explainer in self.explainers:
            cfs = explainer.explain(self.model, query_instance) # must be implemented in explainers class
            all_counterfactuals = pd.concat([all_counterfactuals, cfs], ignore_index=True)

        # those may be put in some HYPERPARAMETERS input dict
        k_neigh_feasibility=3
        k_neigh_discriminative = 9

        train_preds = self.model.predict(data)

        # for now it calculates scores for all counterfactuals 
        # without distingusihing what data point are they explaining
        scores = get_scores(
            cfs=all_counterfactuals.drop(columns=['target']).to_numpy().astype('<U11'),
            cf_predicted_classes=all_counterfactuals['target'].to_numpy(),
            training_data=data.data,
            training_data_predicted_classes=train_preds,
            x = query_instance.data,
            continous_indices=data.continuous_features_ids,
            categorical_indices=data.categorical_features_ids,
            k_neighbors_feasib=k_neigh_feasibility, 
            k_neighbors_discriminative=k_neigh_discriminative
            ).reset_index(drop=True)
        
        filtered_counterfactuals = self.aggregator(all_counterfactuals, scores)
        print(filtered_counterfactuals)
        return filtered_counterfactuals

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

    def explain(self, data: Dataset) -> list[Counterfactual]:
        """This method is used to generate counterfactuals"""

        explain_chains = []
        for explainer in self.explainers:
            explain_chains.append(explainer.explain_async(self.model, data))

        explain_chord = group(explain_chains) | app.signature('ensemble.collect_results')
        results = explain_chord.apply_async().get()

        counterfactuals = [Counterfactual.deserialize(counterfactual) for result in results for counterfactual in result['counterfactuals']]

        return self.aggregator(counterfactuals)
