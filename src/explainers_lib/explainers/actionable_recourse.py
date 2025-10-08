import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Dict, List

import recourse as rs
from lime.lime_tabular import LimeTabularExplainer
from explainers_lib.counterfactual import Counterfactual
from explainers_lib.datasets import Dataset
from explainers_lib.explainers import Explainer
from explainers_lib.model import Model
import recourse.flipset as flipset

# For flipset to work
def patched_add_to_df(self, items):
    if len(items) > 0:
        row_data = list(map(lambda item: self._item_to_df_row(item), items))
        self._df = pd.concat([self._df, pd.DataFrame(row_data)], ignore_index=True)[self._df.columns.tolist()]
        self.sort()

flipset.Flipset._add_to_df = patched_add_to_df


class ActionableRecourseExplainer(Explainer):
    """
    Actionable Recourse explainer adapted from CARLA (Ustun et al., 2019)
    - Works for linear or locally linear approximations (via LIME)
    - Generates actionable counterfactuals given a model and dataset

    Reference:
        Berk Ustun, Alexander Spangher, and Yang Liu. (2019)
        "Actionable Recourse in Linear Classification"
        FAT* '19: Conference on Fairness, Accountability, and Transparency
    """

    def __init__(
        self,
        fs_size: int = 1000,
        discretize: bool = False,
        sample: bool = True,
        coeffs: Optional[np.ndarray] = None,
        intercepts: Optional[np.ndarray] = None,
    ):
        """
        Args:
            fs_size: Number of candidate flip actions to consider
            discretize: Whether to discretize continuous features (for LIME)
            sample: Whether to sample around the instance (for LIME)
            coeffs: Predefined global coefficients (optional)
            intercepts: Predefined global intercepts (optional)
        """
        self.fs_size = fs_size
        self.discretize = discretize
        self.sample = sample
        self.coeffs = coeffs
        self.intercepts = intercepts
        self.source_df = None
        self.action_set = None

    def __repr__(self) -> str:
        return f"actionable_recourse(flip_size={repr(self.fs_size)})"

    def fit(self, model: Model, data: Dataset) -> None:
        """
        Prepare data and initialize action set.
        """
        self.source_df = pd.DataFrame(data.data, columns=data.features)
        self.feature_names = data.features
        self.immutables = getattr(data, "immutable_features", [])
        self.action_set = rs.ActionSet(X=self.source_df[self.feature_names])

        # Mark immutable features as non-actionable
        for feature in self.immutables:
            if feature in self.action_set:
                self.action_set[feature].mutable = False
                self.action_set[feature].actionable = False

    def _get_lime_coefficients(self, model: Model, factuals: pd.DataFrame):
        """
        Generate local linear coefficients using LIME for non-linear models.
        """
        lime_exp = LimeTabularExplainer(
            training_data=self.source_df[self.feature_names].values,
            feature_names=self.feature_names,
            discretize_continuous=self.discretize,
            sample_around_instance=self.sample,
        )

        coeffs = np.zeros((len(factuals), len(self.feature_names)))
        intercepts = []

        for i, (_, row) in enumerate(factuals.iterrows()):
            # reshape row to 2D (1, num_features)
            row_2d = row.values.reshape(1, -1)

            explanation = lime_exp.explain_instance(
                row_2d[0],  # LIME still expects 1D array here
                lambda x: model.predict_proba(x),  # pass 2D array to predict_proba
                num_features=len(self.feature_names),
            )

            intercepts.append(explanation.intercept[1])
            for idx, weight in explanation.local_exp[1]:
                coeffs[i, idx] = weight

        return coeffs, np.array(intercepts)

    def _generate_counterfactual(
        self, instance_ds: Dataset, model: Model, target_class: int
    ) -> Optional[Counterfactual]:
        """
        Generate a counterfactual explanation using Actionable Recourse.
        """
        instance = pd.DataFrame(instance_ds.data, columns=self.feature_names)
        factual = instance.iloc[0]

        # Generate coefficients (via LIME if needed)
        if self.coeffs is None or self.intercepts is None:
            coeffs, intercepts = self._get_lime_coefficients(model, instance)
            coeff, intercept = coeffs[0], intercepts[0]
        else:
            coeff, intercept = self.coeffs, self.intercepts

        # Align action set
        self.action_set.set_alignment(coefficients=coeff)

        # Construct the flipset
        fs = rs.Flipset(
            x=factual.values,
            action_set=self.action_set,
            coefficients=coeff,
            intercept=intercept,
        )

        try:
            fs_pop = fs.populate(total_items=self.fs_size)
        except (ValueError, KeyError):
            return None

        # Iterate through possible actions until one flips the prediction
        for action in fs_pop.actions:
            candidate = factual.values + action
            pred_cf = np.argmax(model.predict_proba(candidate.reshape(1, -1)))
            pred_orig = np.argmax(model.predict_proba(factual.values.reshape(1, -1)))
            if pred_cf != pred_orig and pred_cf == target_class:
                return Counterfactual(
                    original_data=factual.values,
                    data=candidate,
                    original_class=pred_orig,
                    target_class=pred_cf,
                    explainer=repr(self)
                )

        return None

    def explain(
        self, model: Model, data: Dataset, y_desired: Optional[int] = None
    ) -> List[Counterfactual]:
        """
        Generate counterfactuals for all instances in a dataset.
        """
        counterfactuals = []

        for instance in tqdm(data, unit="instance"):
            original_class = model.predict(instance)[0]
            target_classes = (
                [y_desired] if y_desired is not None else range(len(set(data.target)))
            )

            for target_class in target_classes:
                if target_class == original_class:
                    continue

                cf = self._generate_counterfactual(instance, model, target_class)
                if cf is not None:
                    counterfactuals.append(cf)
                    break

        return counterfactuals

    def explain_instance(
        self, instance_ds: Dataset, model: Model, target_class: Optional[int] = None
    ) -> Optional[Counterfactual]:
        """
        Generate a single counterfactual for a specific instance.
        """
        original_class = model.predict(instance_ds)[0]
        if target_class == original_class:
            raise ValueError("Target class cannot be the same as original class")

        if target_class is not None:
            return self._generate_counterfactual(instance_ds, model, target_class)

        # Try all classes if no target specified
        num_classes = getattr(instance_ds, "num_classes", 2)
        for c in range(num_classes):
            if c == original_class:
                continue
            cf = self._generate_counterfactual(instance_ds, model, c)
            if cf is not None:
                return cf

        return None
