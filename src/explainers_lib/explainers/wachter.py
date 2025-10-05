import pandas as pd
import numpy as np
from typing import Optional
from tqdm import tqdm

from explainers_lib.counterfactual import Counterfactual
from explainers_lib.datasets import Dataset
from explainers_lib.explainers import Explainer
from explainers_lib.model import Model

# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import warnings
from explainers_lib.datasets import Dataset

import numpy as np
from scipy.optimize import minimize


def create_counterfactual(
    x_reference,
    y_desired,
    model,
    X_dataset,
    y_desired_proba=None,
    lammbda=0.1,
    random_seed=None,
):
    """
    Implementation of the counterfactual method by Wachter et al. 2017

    References:

    - Wachter, S., Mittelstadt, B., & Russell, C. (2017).
    Counterfactual explanations without opening the black box:
     Automated decisions and the GDPR. Harv. JL & Tech., 31, 841.,
     https://arxiv.org/abs/1711.00399

    Parameters
    ----------

    x_reference : array-like, shape=[m_features]
        The data instance (training example) to be explained.

    y_desired : int
        The desired class label for `x_reference`.

    model : estimator
        A (scikit-learn) estimator implementing `.predict()` and/or
        `predict_proba()`.
        - If `model` supports `predict_proba()`, then this is used by
        default for the first loss term,
        `(lambda * model.predict[_proba](x_counterfact) - y_desired[_proba])^2`
        - Otherwise, method will fall back to `predict`.

    X_dataset : array-like, shape=[n_examples, m_features]
        A (training) dataset for picking the initial counterfactual
        as initial value for starting the optimization procedure.

    y_desired_proba : float (default: None)
        A float within the range [0, 1] designating the desired
        class probability for `y_desired`.
        - If `y_desired_proba=None` (default), the first loss term
        is `(lambda * model(x_counterfact) - y_desired)^2` where `y_desired`
        is a class label
        - If `y_desired_proba` is not None, the first loss term
        is `(lambda * model(x_counterfact) - y_desired_proba)^2`

    lammbda : Weighting parameter for the first loss term,
        `(lambda * model(x_counterfact) - y_desired[_proba])^2`

    random_seed : int (default=None)
        If int, random_seed is the seed used by
        the random number generator for selecting the inital counterfactual
        from `X_dataset`.

    """
    if y_desired_proba is not None:
        use_proba = True
        if not hasattr(model, "predict_proba"):
            raise AttributeError(
                "Your `model` does not support "
                "`predict_proba`. Set `y_desired_proba` "
                " to `None` to use `predict`instead."
            )
    else:
        use_proba = False

    if y_desired_proba is None:
        # class label
        y_to_be_annealed_to = y_desired
    else:
        # class proba corresponding to class label y_desired
        y_to_be_annealed_to = y_desired_proba

    # start with random counterfactual
    rng = np.random.RandomState(random_seed)
    x_counterfact = X_dataset[rng.randint(X_dataset.shape[0])]

    # compute median absolute deviation
    mad = np.abs(np.median(X_dataset, axis=0) - x_reference) + 1e-8

    def dist(x_reference, x_counterfact):
        numerator = np.abs(x_reference - x_counterfact)
        return np.sum(numerator / mad)

    def loss(x_counterfact, lammbda):
        if use_proba:
            y_predict = model.predict_proba(x_counterfact.reshape(1, -1)).flatten()[
                y_desired
            ]
        else:
            y_predict = model.predict(
                Dataset(x_counterfact.reshape(1, -1), [], [], [], [], [], [])
            )[0]

        diff = lammbda * (y_predict - y_to_be_annealed_to) ** 2

        return diff + dist(x_reference, x_counterfact)

    res = minimize(loss, x_counterfact, args=(lammbda), method="Nelder-Mead")

    if not res["success"]:
        warnings.warn(res["message"])

    x_counterfact = res["x"]

    return x_counterfact


# end of Raschka


class WachterExplainer(Explainer):
    def __init__(
        self,
        lambda_param: float = 0.1,
        random_seed: Optional[int] = None,
    ):
        """
        Wachter method explainer using customized mlxtend's create_counterfactual.
        Description: https://rasbt.github.io/mlxtend/user_guide/evaluate/create_counterfactual/

        Args:
            lambda_param: Regularization parameter for distance penalty
            random_seed: Random seed for reproducibility
        """
        self.lambda_param = lambda_param
        self.random_seed = random_seed
        self.source_df = None

    def fit(self, model: Model, data: Dataset) -> None:
        self.source_df = pd.DataFrame(data.data)
        self.source_df

    def explain(
        self, model: Model, data: Dataset, y_desired: int = None
    ) -> list[Counterfactual]:
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

    def _generate_counterfactual(
        self,
        instance_ds: Dataset,
        model: Model,
        target_class: int,
    ) -> Optional[Counterfactual]:
        """
        Generate a counterfactual using the Wachter method.
        """
        instance = instance_ds.data[0]
        original_class = model.predict(instance_ds)[0]

        try:
            counterfactual_data = create_counterfactual(
                x_reference=instance,
                y_desired=target_class,
                model=model,
                X_dataset=self.source_df.values,
                lammbda=self.lambda_param,
                random_seed=self.random_seed,
            )

            cf_ds = instance_ds.like(counterfactual_data.reshape(1, -1))
            predicted_class = model.predict(cf_ds)[0]

            if predicted_class != target_class:
                raise ValueError(
                    "Generated counterfactual does not produce target class"
                )

            return Counterfactual(
                original_data=instance,
                data=counterfactual_data,
                original_class=original_class,
                target_class=predicted_class,
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
