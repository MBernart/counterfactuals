import pandas as pd
import numpy as np
from abc import ABC, abstractmethod  # proposed by gpt
from typing import Callable, List, Optional
from .utils.scores import get_scores
from .model import Model
from .datasets import Dataset
from .counterfactual import Counterfactual
from .utils.pareto import get_pareto_optimal_mask, get_ideal_point
from sklearn.metrics import pairwise_distances


# Aggregator: TypeAlias = Callable[[Sequence[Counterfactual]], Counterfactual]


# Type alias (for convenience or registration)
Aggregator = Callable[[List[Counterfactual], bool], List[Counterfactual]]


class AggregatorBase(ABC):
    """Abstract base class for counterfactual aggregators"""

    def __init__(self):
        self.model = None
        self.data = None
        self.train_preds = None

    def fit(self, model: Model, data: Dataset) -> None:
        """Fits the aggregator by storing model and training data predictions."""
        self.model = model
        self.data = data
        self.train_preds = self.model.predict(self.data)

    def calculate_scores(self, cfs: List[Counterfactual], k_neighbors_feasib: int = 3, k_neighbors_discriminative: int = 9) -> pd.DataFrame:
        """Calculates the standard scores for a list of counterfactuals."""
        if self.model is None or self.data is None:
            raise RuntimeError("Aggregator must be fitted before calculating scores. Call fit(model, data) first.")

        cfs_data_ohe = np.array([cf.data for cf in cfs])
        cfs_df_raw = self.data.inverse_transform(cfs_data_ohe)
        cfs_data_raw = cfs_df_raw.to_numpy()

        # note: must be 2D (1, N) because scores.py slices it like x[:, indices]
        original_data_ohe = cfs[0].original_data.reshape(1, -1)
        original_df_raw = self.data.inverse_transform(original_data_ohe)
        original_data_raw = original_df_raw.to_numpy() 

        training_data_raw = self.data.df.to_numpy()

        feature_names = self.data.features
        
        cont_indices = [feature_names.index(col) for col in self.data.continuous_features]
        cat_indices = [feature_names.index(col) for col in self.data.categorical_features]

        cfs_target = np.array([cf.target_class for cf in cfs])

        return get_scores(
            cfs=cfs_data_raw,
            cf_predicted_classes=cfs_target,
            x=original_data_raw,
            training_data=training_data_raw,
            training_data_predicted_classes=self.train_preds,
            continous_indices=cont_indices,
            categorical_indices=cat_indices,
            k_neighbors_feasib=k_neighbors_feasib,
            k_neighbors_discriminative=k_neighbors_discriminative,
        ).reset_index(drop=True)

    def _attach_scores(self, cfs: List[Counterfactual], scores: pd.DataFrame):
        """Helper to attach diagnostics scores to counterfactual metadata."""
        for i, cf in enumerate(cfs):
            cf.metadata["scores"] = scores.iloc[i].to_dict()

    @abstractmethod
    def __call__(self, cfs: List[Counterfactual], include_scores: bool = False) -> List[Counterfactual]:
        pass


class ScoreBasedAggregator(AggregatorBase):
    """
    Abstract base class for aggregators that rely on a common set of
    scores (Proximity, Feasibility, DiscriminativePower).
    """

    def __init__(self, k_neigh_feasibility: int = 3, k_neigh_discriminative: int = 9):
        super().__init__()
        self.k_neigh_feasibility = k_neigh_feasibility
        self.k_neigh_discriminative = k_neigh_discriminative

    def calculate_scores(self, cfs: List[Counterfactual]) -> pd.DataFrame:
        """Calculates the standard scores using instance parameters."""
        return super().calculate_scores(
            cfs, 
            k_neighbors_feasib=self.k_neigh_feasibility, 
            k_neighbors_discriminative=self.k_neigh_discriminative
        )

    @abstractmethod
    def __call__(self, cfs: List[Counterfactual], include_scores: bool = False) -> List[Counterfactual]:
        pass


class Pareto(ScoreBasedAggregator):
    """Computes the Pareto front from counterfactuals"""

    def __init__(self, k_neigh_feasibility=3, k_neigh_discriminative=9):
        super().__init__(
            k_neigh_feasibility=k_neigh_feasibility,
            k_neigh_discriminative=k_neigh_discriminative,
        )

    def __call__(self, cfs: List[Counterfactual], include_scores: bool = False) -> List[Counterfactual]:
        if not cfs:
            return []
            
        scores = self.calculate_scores(cfs)

        # Example: return all Pareto-efficient counterfactuals
        x_metric = "Proximity"
        y_metric = f"K_Feasibility({self.k_neigh_feasibility})"
        z_metric = f"DiscriminativePower({self.k_neigh_discriminative})"
        optimization_direction = ["min", "min", "max"]

        all_x = scores[x_metric].to_numpy()
        all_y = scores[y_metric].to_numpy()
        all_z = scores[z_metric].to_numpy()
        to_check = np.array([all_x, all_y, all_z], dtype=np.float64).T

        pareto_mask = get_pareto_optimal_mask(
            data=to_check, optimization_direction=optimization_direction
        ).astype(bool)

        pareto_indices = np.where(pareto_mask)[0]
        selected_cfs = [cfs[i] for i in pareto_indices]

        if include_scores:
            # Attach scores to the selected counterfactuals
            for i, idx in enumerate(pareto_indices):
                selected_cfs[i].metadata["scores"] = scores.iloc[idx].to_dict()

        return selected_cfs


class IdealPoint(ScoreBasedAggregator):
    """Computes the ideal point from counterfactuals"""

    def __init__(
        self,
        weights: List[float] = None,
        k_neigh_feasibility=3,
        k_neigh_discriminative=9,
    ):
        """
        weights: optional list of 3 weights for x, y, z metrics
        (will be normalized to sum = 1). If None, equal weights are used.
        """
        super().__init__(
            k_neigh_feasibility=k_neigh_feasibility,
            k_neigh_discriminative=k_neigh_discriminative,
        )
        self.weights = weights

    def __call__(self, cfs: List[Counterfactual], include_scores: bool = False) -> List[Counterfactual]:
        if not cfs:
            return []

        scores = self.calculate_scores(cfs)

        x_metric = "Proximity"
        y_metric = f"K_Feasibility({self.k_neigh_feasibility})"
        z_metric = f"DiscriminativePower({self.k_neigh_discriminative})"
        optimization_direction = ["min", "min", "max"]

        all_x = scores[x_metric].to_numpy()
        all_y = scores[y_metric].to_numpy()
        all_z = scores[z_metric].to_numpy()
        to_check = np.array([all_x, all_y, all_z], dtype=np.float64).T

        pareto_mask = get_pareto_optimal_mask(
            data=to_check, optimization_direction=optimization_direction
        ).astype(bool)

        pareto_indices = np.where(pareto_mask)[0]
        pareto_cfs = [cfs[i] for i in pareto_indices]
        pareto_data = to_check[pareto_mask]

        ideal_point = get_ideal_point(to_check, optimization_direction, pareto_mask)

        if self.weights is None:
            weights = np.ones(to_check.shape[1]) / to_check.shape[1]
        else:
            weights = np.array(self.weights, dtype=float)
            weights = weights / weights.sum()  # normalize

        # weighted distances
        diffs = pareto_data - ideal_point
        dists = np.sqrt(np.sum(weights * diffs**2, axis=1))
        # pick closest
        best_pareto_idx = np.argmin(dists)
        best_idx = pareto_indices[best_pareto_idx]
        
        selected_cfs = [cfs[best_idx]]

        if include_scores:
            selected_cfs[0].metadata["scores"] = scores.iloc[best_idx].to_dict()

        return selected_cfs


class BalancedPoint(ScoreBasedAggregator):
    """
    Selects a single counterfactual closest to the midpoint between the ideal and nadir points
    (i.e., a balanced trade-off solution).
    """

    def __init__(self, k_neigh_feasibility=3, k_neigh_discriminative=9):
        super().__init__(
            k_neigh_feasibility=k_neigh_feasibility,
            k_neigh_discriminative=k_neigh_discriminative,
        )

    def __call__(self, cfs: List[Counterfactual], include_scores: bool = False) -> List[Counterfactual]:
        if not cfs:
            return []

        scores = self.calculate_scores(cfs)

        x_metric = "Proximity"
        y_metric = f"K_Feasibility({self.k_neigh_feasibility})"
        z_metric = f"DiscriminativePower({self.k_neigh_discriminative})"
        optimization_direction = ["min", "min", "max"]

        all_x = scores[x_metric].to_numpy()
        all_y = scores[y_metric].to_numpy()
        all_z = scores[z_metric].to_numpy()
        to_check = np.array([all_x, all_y, all_z], dtype=np.float64).T

        pareto_mask = get_pareto_optimal_mask(
            data=to_check, optimization_direction=optimization_direction
        ).astype(bool)

        pareto_indices = np.where(pareto_mask)[0]
        pareto_data = to_check[pareto_mask]

        # Compute ideal and nadir points
        ideal_point = get_ideal_point(to_check, optimization_direction, pareto_mask)

        nadir_point = np.array(
            [
                np.max(pareto_data[:, i]) if opt == "min" else np.min(pareto_data[:, i])
                for i, opt in enumerate(optimization_direction)
            ],
            dtype=np.float64,
        )

        # Midpoint between ideal and nadir (balanced region)
        midpoint = (ideal_point + nadir_point) / 2

        # Choose Pareto solution closest to midpoint
        dists = np.linalg.norm(pareto_data - midpoint, axis=1)
        best_pareto_idx = np.argmin(dists)
        best_idx = pareto_indices[best_pareto_idx]

        selected_cfs = [cfs[best_idx]]

        if include_scores:
            selected_cfs[0].metadata["scores"] = scores.iloc[best_idx].to_dict()

        return selected_cfs


class ParetoMeanPoint(ScoreBasedAggregator):
    """
    Selects a single counterfactual from the Pareto front that is closest
    to the mean (or median) of all points on the Pareto front.
    """

    def __init__(self, metric: str = "mean", k_neigh_feasibility=3, k_neigh_discriminative=9):
        super().__init__(
            k_neigh_feasibility=k_neigh_feasibility,
            k_neigh_discriminative=k_neigh_discriminative,
        )
        self.metric = metric

    def __call__(self, cfs: List[Counterfactual]) -> List[Counterfactual]:
        if not cfs:
            return []

        scores = self.calculate_scores(cfs)

        x_metric = "Proximity"
        y_metric = f"K_Feasibility({self.k_neigh_feasibility})"
        z_metric = f"DiscriminativePower({self.k_neigh_discriminative})"
        optimization_direction = ["min", "min", "max"]

        all_x = scores[x_metric].to_numpy()
        all_y = scores[y_metric].to_numpy()
        all_z = scores[z_metric].to_numpy()
        to_check = np.array([all_x, all_y, all_z], dtype=np.float64).T

        pareto_mask = get_pareto_optimal_mask(
            data=to_check, optimization_direction=optimization_direction
        ).astype(bool)

        pareto_indices = np.where(pareto_mask)[0]
        if len(pareto_indices) == 0:
            return []

        pareto_cfs = [cfs[i] for i in pareto_indices]
        pareto_data = to_check[pareto_mask]

        if self.metric == "mean":
            center_point = np.mean(pareto_data, axis=0)
        elif self.metric == "median":
            center_point = np.median(pareto_data, axis=0)
        else:
            raise ValueError(f"Unknown metric: {self.metric}. Use 'mean' or 'median'.")

        dists = np.linalg.norm(pareto_data - center_point, axis=1)
        best_idx = np.argmin(dists)

        return [pareto_cfs[best_idx]]


class TOPSIS(ScoreBasedAggregator):
    """
    Selects the top-k counterfactuals using the TOPSIS
    (Technique for Order of Preference by Similarity to Ideal Solution) method.
    <see: https://www.researchgate.net/publication/285886027_Notes_on_TOPSIS_Method>
    """

    def __init__(
        self,
        k: int = 1,
        k_neigh_feasibility=3,
        k_neigh_discriminative=9,
    ):
        """
        Parameters
        ----------
        k : int
            Number of top counterfactuals to return.
        """
        super().__init__(
            k_neigh_feasibility=k_neigh_feasibility,
            k_neigh_discriminative=k_neigh_discriminative,
        )
        self.weights = None
        self.k = k

    def __call__(self, cfs: List[Counterfactual], include_scores: bool = False) -> List[Counterfactual]:
        if not cfs or len(cfs) < 1:
            return []

        scores = self.calculate_scores(cfs)
        if scores.empty:
            return []

        criteria_cols = [
            "Proximity",
            f"K_Feasibility({self.k_neigh_feasibility})",
            f"DiscriminativePower({self.k_neigh_discriminative})",
        ]
        optimization_direction = ["min", "min", "max"]

        matrix = scores[criteria_cols].to_numpy(dtype=np.float64)

        if self.weights is None:
            weights = np.ones(matrix.shape[1]) / matrix.shape[1]
        else:
            if len(self.weights) != matrix.shape[1]:
                raise ValueError(
                    f"List of weight's length ({len(self.weights)}) must "
                    f"match number of criteria ({matrix.shape[1]})"
                )
            weights = np.array(self.weights, dtype=float)
            weights = weights / weights.sum()

        norms = np.linalg.norm(matrix, axis=0)
        norms[norms == 0] = np.finfo(float).eps
        normalized_matrix = matrix / norms

        weighted_matrix = normalized_matrix * weights

        col_maxes = np.max(weighted_matrix, axis=0)
        col_mins = np.min(weighted_matrix, axis=0)

        is_max_direction = np.array(optimization_direction) == "max"

        ideal_solution = np.where(is_max_direction, col_maxes, col_mins)
        anti_ideal_solution = np.where(is_max_direction, col_mins, col_maxes)

        dist_to_ideal = np.linalg.norm(weighted_matrix - ideal_solution, axis=1)
        dist_to_anti_ideal = np.linalg.norm(
            weighted_matrix - anti_ideal_solution, axis=1
        )

        denominator = dist_to_ideal + dist_to_anti_ideal
        denominator[denominator == 0] = np.finfo(float).eps
        closeness_score = dist_to_anti_ideal / denominator

        ranked_indices = np.argsort(closeness_score)[::-1]

        top_k_indices = ranked_indices[: self.k]

        selected_cfs = [cfs[i] for i in top_k_indices]

        if include_scores:
            for i, idx in enumerate(top_k_indices):
                selected_cfs[i].metadata["scores"] = scores.iloc[idx].to_dict()

        return selected_cfs


class DensityBased(AggregatorBase):
    """Selects k diverse-yet-similar counterfactuals using a density-based objective (Cost Scaled Greedy)."""

    def __init__(self, k: int = 5, n: int = 3, lambda_: float = 0.5, metric: str = "euclidean"):
        """
        Parameters
        ----------
        k : int
            Number of counterfactuals to select.
        n : int
            Number of nearest neighbors to consider in diversity term.
        lambda_ : float
            Regularization balancing diversity vs. similarity (higher favors similarity).
        metric : str
            Distance metric for pairwise and instance comparisons.
        """
        super().__init__()
        self.k = k
        self.n = n
        self.lambda_ = lambda_
        self.metric = metric

    def _compute_knn_sets(self, cf_data: np.ndarray) -> List[set]:
        """Compute sets of k-nearest neighbors (indices) for each counterfactual."""
        dists = pairwise_distances(cf_data, metric=self.metric)
        np.fill_diagonal(dists, np.inf)
        knn_sets = [set(np.argsort(dists[i])[:self.n]) for i in range(len(cf_data))]
        return knn_sets, dists

    def __call__(self, cfs: List[Counterfactual], include_scores: bool = False) -> List[Counterfactual]:
        if not cfs:
            return []
            
        if len(cfs) <= self.k:
            selected_indices = list(range(len(cfs)))
        else:
            cf_data = np.array([cf.data for cf in cfs])
            original_x = cfs[0].original_data.reshape(1, -1)

            # Compute knn sets and pairwise distances
            knn_sets, pairwise_dists = self._compute_knn_sets(cf_data)
            dist_to_x = pairwise_distances(cf_data, original_x, metric=self.metric).flatten()

            selected_indices = []
            covered = set()

            for _ in range(self.k):
                best_cf_idx = None
                best_gain = -np.inf

                for i in range(len(cfs)):
                    if i in selected_indices:
                        continue

                    # Coverage gain (new neighbors added)
                    new_coverage = len(knn_sets[i] - covered)
                    similarity_penalty = self.lambda_ * dist_to_x[i]
                    gain = new_coverage - similarity_penalty

                    if gain > best_gain:
                        best_gain = gain
                        best_cf_idx = i

                if best_cf_idx is None or best_gain <= 0:
                    break  # No improvement
                selected_indices.append(best_cf_idx)
                covered.update(knn_sets[best_cf_idx])

        selected_cfs = [cfs[i] for i in selected_indices]
        
        if include_scores:
            scores = self.calculate_scores(cfs)
            for i, idx in enumerate(selected_indices):
                selected_cfs[i].metadata["scores"] = scores.iloc[idx].to_dict()
                
        return selected_cfs


class All(AggregatorBase):
    """Return all (valid) counterfactuals found by explainer"""

    def __call__(self, cfs: List[Counterfactual], include_scores: bool = False) -> List[Counterfactual]:
        if include_scores and cfs:
            scores = self.calculate_scores(cfs)
            self._attach_scores(cfs, scores)
        return cfs
