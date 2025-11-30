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
        self.data_interface = None
        self.feature_names = None
        self.preprocessor = None 
        self.feature_dtypes = None

    def __repr__(self):
        return f"dice_explainer(num_cfs={self.num_cfs}, method='{self.method}')"

    def fit(self, model: Model, data: Dataset):
        """
        Prepare DiCE with given model and dataset. 
        Uses the RAW dataframe so DiCE can handle categoricals correctly.
        """
        df = data.df.copy()
        
        cols_to_fix = [col for col in data.continuous_features if col in df.columns]
        
        if cols_to_fix:
            df[cols_to_fix] = df[cols_to_fix].astype("float64")

        target_col = "target"
        target_df = pd.DataFrame({target_col: data.target}, index=df.index)
        df = pd.concat([df, target_df], axis=1)

        self.feature_dtypes = data.df[data.features].dtypes

        self.data_interface = dice_ml.Data(
            dataframe=df,
            continuous_features=data.continuous_features,
            outcome_name=target_col,
        )

        self.preprocessor = data.preprocessor
        self.feature_names = data.features

        model_wrapper = _ModelPipelineWrapper(model, data.preprocessor, self.feature_dtypes)
        self.model = dice_ml.Model(model=model_wrapper, backend="sklearn")

        self.dice = dice_ml.Dice(self.data_interface, self.model, method=self.method)

    def explain(self, model: Model, data: Dataset, y_desired: Optional[int] = None) -> List[Counterfactual]:
        """
        Generate counterfactuals for multiple instances.
        """
        counterfactuals = []
        y_target = y_desired or self.desired_class or 1
        
        df_raw = data.df.reset_index(drop=True)
        for col in data.continuous_features:
            if col in df_raw.columns:
                df_raw[col] = df_raw[col].astype("float64")

        for i in tqdm(range(len(df_raw)), unit="instance"):
            instance_df = df_raw.iloc[[i]]
            
            original_vector = data.data[i]
            
            cf = self._generate_cf(instance_df, original_vector, model, y_target)
            if cf is not None:
                counterfactuals.append(cf)

        return counterfactuals

    def explain_instance(
        self, instance_ds: Dataset, model: Model, target_class: Optional[int] = None
    ) -> Optional[Counterfactual]:
        """
        Generate counterfactual for a single instance.
        """
        instance_df = instance_ds.df.copy()
        original_vector = instance_ds.data[0]
        target = target_class or self.desired_class or 1
        return self._generate_cf(instance_df, original_vector, model, target)

    def _generate_cf(
        self, 
        instance_df: pd.DataFrame, 
        original_vector: np.ndarray, 
        model: Model, 
        target_class: int
    ) -> Optional[Counterfactual]:
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
                candidate_raw_df = df_cfs[self.feature_names].iloc[[0]].copy()
                
                for col, dtype in self.feature_dtypes.items():
                    if col in candidate_raw_df.columns:
                        candidate_raw_df[col] = candidate_raw_df[col].astype(dtype)

                candidate_vector = self.preprocessor.transform(candidate_raw_df).flatten()
                
                pred_orig = np.argmax(model.predict_proba(original_vector.reshape(1, -1)))
                pred_cf = np.argmax(model.predict_proba(candidate_vector.reshape(1, -1)))

                return Counterfactual(
                    original_data=original_vector,
                    data=candidate_vector,
                    original_class=pred_orig,
                    target_class=pred_cf,
                    explainer=repr(self),
                )

        except Exception as e:
            print(f"[WARN] DiCE failed for instance: {e}")

        return None

class _ModelPipelineWrapper:
    """
    Internal helper to glue the Preprocessor and the Model together.
    Enforces strict data types to prevent OHE errors.
    """
    def __init__(self, model_obj, preprocessor, feature_dtypes):
        self.model = model_obj
        self.preprocessor = preprocessor
        self.feature_dtypes = feature_dtypes

    def _ensure_dtypes(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        DiCE often converts data to pure strings or floats. 
        We must cast them back to the training dtypes (e.g. int64) 
        so the OneHotEncoder recognizes the categories.
        """
        df_fixed = dataframe.copy()
        
        numeric_cols = [c for c, d in self.feature_dtypes.items() if np.issubdtype(d, np.number)]
        object_cols = [c for c, d in self.feature_dtypes.items() if d == object or str(d) == 'object']
        
        if numeric_cols:
            valid_num = [c for c in numeric_cols if c in df_fixed.columns]
            if valid_num:
                df_fixed[valid_num] = df_fixed[valid_num].astype("float64")
        
        if object_cols:
            valid_obj = [c for c in object_cols if c in df_fixed.columns]
            if valid_obj:
                df_fixed[valid_obj] = df_fixed[valid_obj].astype(str)

        return df_fixed

    def predict_proba(self, dataframe):
        df_typed = self._ensure_dtypes(dataframe)
        X_transformed = self.preprocessor.transform(df_typed)
        return self.model.predict_proba(X_transformed)

    def predict(self, dataframe):
        df_typed = self._ensure_dtypes(dataframe)
        X_transformed = self.preprocessor.transform(df_typed)
        return self.model.predict(X_transformed)
