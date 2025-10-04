from celery import group
import pandas as pd
from .model import Model
from .explainers import Explainer
from .explainers.celery_remote import app
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
        self, model_data: bytes, explainers: list[str], aggregator: Aggregator = All
    ):
        """Constructs an ensemble"""

        self.model_data = model_data
        self.aggregator = aggregator

        available_explainers = app.send_task('ensemble.get_explainers').get()
        # TODO: Need to check that selected explainers are actually available (ex. ping)
        self.explainers = list(filter(lambda explainer: explainer in available_explainers, explainers))
        missing_explainers = list(filter(lambda explainer: explainer not in available_explainers, explainers))

        if len(self.explainers) == 0:
            raise RuntimeError(f"CeleryEnsemble: Explainers not found: {explainers}")
        
        if len(missing_explainers) > 0:
            raise RuntimeWarning(f"CeleryEnsemble: Explainers not found: {missing_explainers}")

    def fit(self, data: Dataset) -> None:
        """This method is used to train all explainers in the ensemble"""

        fit_chains = []
        for explainer_name in self.explainers:

            set_dataset_sig = app.signature(
                f'{explainer_name}.set_dataset',
                args=[data.serialize()],
                queue=explainer_name
            )

            set_model_sig = app.signature(
                f'{explainer_name}.set_model',
                args=[self.model_data, 'torch'],
                queue=explainer_name
            )

            fit_sig = app.signature(
                f'{explainer_name}.fit',
                queue=explainer_name
            )

            fit_chain = group([set_dataset_sig, set_model_sig]) | fit_sig
            fit_chains.append(fit_chain)

        fit_chord = group(fit_chains) | app.signature('ensemble.collect_results')
        fit_chord.apply_async().get()

    def explain(self, data: Dataset) -> list[Counterfactual]:
        """This method is used to generate counterfactuals"""

        explain_chains = []
        for explainer_name in self.explainers:

            set_dataset_sig = app.signature(
                f'{explainer_name}.set_dataset',
                args=[data.serialize()],
                queue=explainer_name
            )

            explain_sig = app.signature(
                f'{explainer_name}.explain',
                queue=explainer_name
            )

            explain_chain = set_dataset_sig | explain_sig
            explain_chains.append(explain_chain)

        explain_chord = group(explain_chains) | app.signature('ensemble.collect_results')
        results = explain_chord.apply_async().get()

        counterfactuals = [Counterfactual.deserialize(counterfactual) for result in results for counterfactual in result['counterfactuals']]

        return self.aggregator(counterfactuals)
