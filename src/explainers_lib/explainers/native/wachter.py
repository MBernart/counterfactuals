import pandas as pd
import numpy as np
from typing import Optional
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
        lambda_param: float = 0.1,
        random_seed: Optional[int] = None,
        max_iter: int = 1000,
        optimization_method: str = "COBYLA",
    ):
        self.lambda_param = lambda_param
        self.random_seed = random_seed
        self.max_iter = max_iter
        self.source_df = None
        self.optimization_method = optimization_method

    def __repr__(self) -> str:
        return (
            f"wachter(lambda_param={repr(self.lambda_param)}, method={self.optimization_method},"
            f"max_iter={self.max_iter}, random_seed={self.random_seed})"
        )

    def fit(self, model: Model, data: Dataset) -> None:
        X_train = data.df if hasattr(data, "df") else data.data
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values

        medians = np.median(X_train, axis=0)
        abs_diff = np.abs(X_train - medians)
        self.mad = np.median(abs_diff, axis=0)

        self.mad[self.mad < 1e-5] = 1.0

        self.bounds = list(zip(X_train.min(axis=0), X_train.max(axis=0)))

    def explain(
        self, model: Model, data: Dataset, y_desired: int = None
    ) -> list[Counterfactual]:
        if isinstance(data, TorchModel):
            model._model.to("cuda")

        counterfactuals = []

        for instance in tqdm(data, unit="instance"):
            original_class = model.predict(instance)[0]

            target_range = (
                [y_desired] if y_desired is not None else range(len(set(data.target)))
            )

            for target_class in target_range:
                if target_class == original_class:
                    continue

                try:
                    cf = self._generate_counterfactual(instance, model, target_class)
                    if cf is not None:
                        counterfactuals.append(cf)
                        break
                except ValueError as e:
                    continue

        return counterfactuals

    def _loss_function(
        self, x_candidate, model, x_original, target_class, lambda_param, mad
    ):
        """
        The objective function to minimize:
        Loss = lambda * (Prob(target) - 1)^2 + Distance(x_candidate, x_original)
        """
        x_candidate = x_candidate.reshape(1, -1)[0]

        probs = model.predict_proba(x_candidate)[0]
        probs = softmax(probs)
        target_prob = probs[target_class]

        pred_loss = (target_prob - 1.0) ** 2

        dist_loss = np.sum(np.abs(x_candidate.flatten() - x_original.flatten()) / mad)

        return lambda_param * pred_loss + dist_loss

    def _generate_counterfactual(
        self,
        instance_ds: Dataset,
        model: Model,
        target_class: int,
    ) -> Optional[Counterfactual]:
        """
        Generate a counterfactual using the Wachter method.
        """

        try:
            original_class = model.predict(instance_ds)[0]
            if target_class == original_class:
                raise ValueError("Target class cannot be the same as original class")
            x_instance = np.array(instance_ds.data[0]).flatten()

            initial_guess = x_instance

            result = minimize(
                self._loss_function,
                initial_guess,
                args=(model, x_instance, target_class, self.lambda_param, self.mad),
                method=self.optimization_method,
                bounds=None,
                options={"maxiter": self.max_iter, "disp": False},
            )

            cf_candidate = result.x.reshape(1, -1)

            if np.argmax(model.predict_proba(cf_candidate)[0]) == target_class:
                return Counterfactual(
                    original_data=np.array(instance_ds.data[0]),
                    data=cf_candidate,
                    original_class=original_class,
                    target_class=np.argmax(model.predict_proba(cf_candidate)[0]),
                    explainer=repr(self),
                )

        except Exception as e:
            raise ValueError(f"Failed to generate counterfactual: {str(e)}")

    def explain_instance(
        self, instance_ds: Dataset, model: Model, target_class: Optional[int] = None
    ) -> Optional[Counterfactual]:
        original_class = model.predict(instance_ds)[0]

        if target_class is not None:
            if target_class == original_class:
                raise ValueError("Target class cannot be the same as original class")
            try:
                return self._generate_counterfactual(instance_ds, model, target_class)
            except ValueError:
                return None
        else:
            num_classes = (
                len(set(instance_ds.target)) if hasattr(instance_ds, "target") else 2
            )
            for target_class in range(num_classes):
                if target_class == original_class:
                    continue

                try:
                    cf = self._generate_counterfactual(instance_ds, model, target_class)
                    if cf is not None:
                        return cf
                except ValueError:
                    continue

            return None
