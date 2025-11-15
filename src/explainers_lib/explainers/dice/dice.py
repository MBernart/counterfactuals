import pandas as pd
import numpy as np
from typing import List, Optional

import dice_ml
from tqdm import tqdm

from explainers_lib.datasets import Dataset
from explainers_lib.explainers import Explainer
from explainers_lib.counterfactual import Counterfactual
from explainers_lib.model import Model


class DiceExplainer(Explainer):
    """
    DiCE-based counterfactual explainer (Mothilal et al., 2020).

    References:
        R. K. Mothilal, Amit Sharma, and Chenhao Tan.
        "Explaining machine learning classifiers through diverse counterfactual explanations" (2020)
    """

    def __init__(
        self,
        num_cfs: int = 1,
        desired_class: Optional[int] = None,
        method: str = "random",
        posthoc_sparsity_param: float = 0.1,
    ):
        """
        Args:
            num_cfs: Number of counterfactuals to generate per instance.
            desired_class: Desired target class for generated counterfactuals.
            method: DiCE generation method ('random', 'genetic', or 'kdtree').
            posthoc_sparsity_param: Fraction for DiCE's post-hoc sparsity.
        """
        self.num_cfs = num_cfs
        self.desired_class = desired_class
        self.method = method
        self.posthoc_sparsity_param = posthoc_sparsity_param
        self.dice = None
        self.data = None
        self.model = None
        self.feature_names = None

    def __repr__(self):
        return f"dice_explainer(num_cfs={self.num_cfs}, method='{self.method}')"

    def fit(self, model: Model, data: Dataset):
        """
        Prepare DiCE with given model and dataset.
        """
        # Convert dataset to dataframe for DiCE
        df = pd.DataFrame(data.data, columns=data.features)
        df["target"] = data.target

        # Build DiCE data + model
        self.data = dice_ml.Data(
            dataframe=df,
            continuous_features=data.continuous_features,
            outcome_name="target",
        )

        # Wrap the CARLA model
        backend = "PYT" if hasattr(model, "_torch") else "sklearn"
        self.model = dice_ml.Model(model=model._model, backend=backend)

        # Initialize DiCE explainer
        self.dice = dice_ml.Dice(self.data, self.model, method=self.method)
        self.feature_names = data.features

    def explain(self, model: Model, data: Dataset, y_desired: Optional[int] = None) -> List[Counterfactual]:
        """
        Generate counterfactuals for multiple instances.
        """
        counterfactuals = []
        df = pd.DataFrame(data.data, columns=data.features)
        y_target = y_desired or self.desired_class or 1

        for i in tqdm(range(len(df)), unit="instance"):
            instance_df = df.iloc[[i]]
            cf = self._generate_cf(instance_df, model, y_target)
            if cf is not None:
                counterfactuals.append(cf)

        return counterfactuals

    def explain_instance(
        self, instance_ds: Dataset, model: Model, target_class: Optional[int] = None
    ) -> Optional[Counterfactual]:
        """
        Generate counterfactual for a single instance.
        """
        instance_df = pd.DataFrame(instance_ds.data, columns=self.feature_names)
        target = target_class or self.desired_class or 1
        return self._generate_cf(instance_df, model, target)

    def _generate_cf(self, instance_df: pd.DataFrame, model: Model, target_class: int) -> Optional[Counterfactual]:
        """
        Helper to generate a single counterfactual using DiCE.
        """
        try:
            dice_exp = self.dice.generate_counterfactuals(
                instance_df,
                total_CFs=self.num_cfs,
                desired_class=target_class,
                posthoc_sparsity_param=self.posthoc_sparsity_param,
            )
            df_cfs = dice_exp.cf_examples_list[0].final_cfs_df

            if not df_cfs.empty:
                original = instance_df[self.feature_names].values[0]
                candidate = df_cfs[self.feature_names].iloc[0].values
                pred_orig = np.argmax(model.predict_proba(original.reshape(1, -1)))
                pred_cf = np.argmax(model.predict_proba(candidate.reshape(1, -1)))
                return Counterfactual(
                    original_data=original,
                    data=candidate,
                    original_class=pred_orig,
                    target_class=pred_cf,
                    explainer=repr(self),
                )

        except Exception as e:
            print(f"[WARN] DiCE failed for instance: {e}")
            return None

        return None
