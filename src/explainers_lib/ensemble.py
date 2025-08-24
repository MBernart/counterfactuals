import pandas as pd
from .model import Model
from .explainers import Explainer
from .aggregators import Aggregator
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
