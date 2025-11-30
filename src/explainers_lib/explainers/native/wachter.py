import numpy as np
from typing import Optional, List, Tuple, Union
from tqdm import tqdm

from explainers_lib.counterfactual import Counterfactual
from explainers_lib.datasets import Dataset
from explainers_lib.explainers import Explainer
from explainers_lib.model import Model, TorchModel
from scipy.optimize import minimize
from scipy.special import softmax


class WachterExplainer(Explainer):
    """
    Implements the Wachter Counterfactual Explanation method based on the paper
    "Counterfactual Explanations without Opening the Black Box: Automated Decisions and the GDPR"
    (Wachter et al., 2017).

    Since it relies on black-box optimization (Nelder-Mead, COBYLA, etc.), it works for any model
    that outputs probabilities or logits, including PyTorch, TensorFlow, and Scikit-Learn models,
    without needing access to gradients.

    Attributes:
        lambda_param (float): The regularization strength. A higher lambda prioritizes changing
            the class label (Prediction Loss) over keeping the instance close to the original
            (Distance Loss). Default is 0.1.
        max_iter (int): Maximum number of iterations for the scipy optimizer. Default is 1000.
        optimization_method (str): The gradient-free optimization method used by scipy.optimize.
            Options include 'COBYLA', 'Nelder-Mead', 'Powell'. Default is 'COBYLA'.
            For more details, see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
        random_seed (int, optional): Seed for reproducibility.
    """

    def __init__(
        self,
        lambda_param: Union[float, List[float]] = 0.1,
        random_seed: Optional[int] = None,
        max_iter: int = 1000,
        optimization_method: str = "COBYLA",
    ):
        self.lambda_param = lambda_param
        self.random_seed = random_seed
        self.max_iter = max_iter
        self.optimization_method = optimization_method
        
        self.cat_indices_groups: List[List[int]] = []
        self.immutable_indices: List[int] = []
        self.mad: np.ndarray = None
        self.feature_bounds: List[Tuple[float, float]] = []

    def __repr__(self) -> str:
        lambda_param = "ADAPTIVE" if isinstance(self.lambda_param, list) else self.lambda_param
        return (
            f"wachter(lambda_param={repr(lambda_param)}, method={self.optimization_method}, "
            f"max_iter={self.max_iter}, random_seed={self.random_seed})"
        )

    def fit(self, model: Model, data: Dataset) -> None:
        X_train = data.data 

        if hasattr(X_train, "values"):
            X_train = X_train.values

        medians = np.median(X_train, axis=0)
        abs_diff = np.abs(X_train - medians)
        self.mad = np.median(abs_diff, axis=0)

        self.mad[self.mad < 1e-5] = 1.0

        self.cat_indices_groups = []
        self.immutable_indices = []

        for i, feat_name in enumerate(data.continuous_features):
            if feat_name in data.immutable_features:
                self.immutable_indices.append(i)

        current_idx = len(data.continuous_features)
        
        for feat_name in data.categorical_features:
            n_cats = len(data.categorical_values[feat_name])
            group_indices = list(range(current_idx, current_idx + n_cats))
            
            self.cat_indices_groups.append(group_indices)
            
            if feat_name in data.immutable_features:
                self.immutable_indices.extend(group_indices)
                
            current_idx += n_cats
            
        self.immutable_indices = np.array(self.immutable_indices, dtype=int)

        self.feature_bounds = []
        for i in range(len(data.continuous_features)):
            col_min = float(X_train[:, i].min())
            col_max = float(X_train[:, i].max())
            self.feature_bounds.append((col_min, col_max))

        total_cols = X_train.shape[1]
        start_cat_idx = len(data.continuous_features)
        
        for _ in range(start_cat_idx, total_cols):
            self.feature_bounds.append((0.0, 1.0))

    def explain(self, model: Model, data: Dataset, y_desired: int = None) -> List[Counterfactual]:
        if isinstance(data, TorchModel):
            model._model.to("cuda")

        counterfactuals = []
        X_data = data.data 

        for i in tqdm(range(len(X_data)), unit="instance"):
            instance_raw = X_data[i]
            original_class = np.argmax(model.predict_proba(instance_raw.reshape(1, -1)))
            target_range = ([y_desired] if y_desired is not None else range(len(set(data.target))))

            for target_class in target_range:
                if target_class == original_class:
                    continue
                try:
                    cf = self._generate_counterfactual(instance_raw, model, target_class, original_class)
                    if cf is not None:
                        counterfactuals.append(cf)
                        break 
                except ValueError:
                    continue

        return counterfactuals

    def _loss_function(
        self, x_candidate, model, x_original, target_class, lambda_param, mad
    ):
        """
        The objective function to minimize:
        Loss = lambda * (Prob(target) - 1)^2 + Distance(x_candidate, x_original)
        """
        if len(self.immutable_indices) > 0:
            x_candidate[self.immutable_indices] = x_original[self.immutable_indices]

        x_cand_reshaped = x_candidate.reshape(1, -1)

        probs = model.predict_proba(x_cand_reshaped)[0]
        probs = softmax(probs)
        target_prob = probs[target_class]

        pred_loss = (target_prob - 1.0) ** 2

        dist_loss = np.sum(np.abs(x_candidate - x_original) / mad)

        return lambda_param * pred_loss + dist_loss

    def _project_to_valid_ohe(self, x_candidate: np.ndarray) -> np.ndarray:
        x_projected = x_candidate.copy()
        for group in self.cat_indices_groups:
            best_idx = np.argmax(x_projected[group])
            x_projected[group] = 0.0
            x_projected[group[best_idx]] = 1.0
        return x_projected

    def _generate_counterfactual(
        self,
        instance_array: np.ndarray,
        model: Model,
        target_class: int,
        original_class: int
    ) -> Optional[Counterfactual]:
        """
        Generate a counterfactual using the Wachter method.
        """

        try:
            if target_class == original_class:
                raise ValueError("Target class cannot be the same as original class")
            
            x_instance = instance_array.flatten()
            initial_guess = x_instance.copy()

            # Adaptive Lambda Search
            # We try progressively stronger penalties for the prediction loss.
            # This forces the optimizer to prioritize 'flipping the class' over 'staying close'.
            lambda_values = self.lambda_param if isinstance(self.lambda_param, list) else [self.lambda_param]
            
            for lambda_param in lambda_values:    
                result = minimize(
                    self._loss_function,
                    initial_guess,
                    args=(model, x_instance, target_class, lambda_param, self.mad),
                    method=self.optimization_method,
                    bounds=self.feature_bounds,
                    options={"maxiter": self.max_iter, "disp": False},
                )

                cf_soft = result.x
                
                if len(self.immutable_indices) > 0:
                    cf_soft[self.immutable_indices] = x_instance[self.immutable_indices]
                
                cf_hard = self._project_to_valid_ohe(cf_soft)
                
                if len(self.immutable_indices) > 0:
                    cf_hard[self.immutable_indices] = x_instance[self.immutable_indices]

                cf_candidate = cf_hard.reshape(1, -1)

                pred_probs = model.predict_proba(cf_candidate)[0]
                pred_class = np.argmax(pred_probs)

                if pred_class == target_class:
                    return Counterfactual(
                        original_data=x_instance,
                        data=cf_candidate.flatten(),
                        original_class=original_class,
                        target_class=pred_class,
                        explainer=repr(self),
                    )
        except Exception as e:
            raise ValueError(f"Failed to generate counterfactual: {str(e)}")
