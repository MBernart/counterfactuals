import numpy as np
import pandas as pd
from typing import List, Optional
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path  # for Dijkstra's

from explainers_lib.datasets import Dataset
from explainers_lib.explainers import Explainer
from explainers_lib.counterfactual import Counterfactual
from explainers_lib.model import Model


class FaceExplainer(Explainer):
    """
    FACE-based counterfactual explainer (Poyiadzi et al., 2020).
    Uses Dijkstra's algorithm on a nearest-neighbor graph to approximate the geodesic 
    distance on the data manifold.
    """

    def __init__(
        self,
        mode: str = "knn",
        fraction: float = 1.0,
        desired_class: Optional[int] = None,
        n_neighbors: int = 50,
    ):
        """
        Args:
            mode: Graph-building mode ("knn" or "epsilon"). Note: Only "knn" implemented here.
            fraction: Fraction of data to construct neighborhood graph.
            desired_class: Desired output class for generated counterfactuals.
            n_neighbors: Number of neighbors for kNN search.
        """
        self.mode = mode
        self.fraction = fraction
        self.desired_class = desired_class
        self.n_neighbors = n_neighbors
        self.model = None
        
        self.X = None
        self.y = None
        self.transformed_feature_names = None
        
        self.neigh = None
        self.dataset_ref = None
        
        self.X_graph = None # The subsetted data used to build the graph
        self.y_graph = None # The corresponding labels
        self.adjacency_matrix = None # The sparse graph for shortest path search

    def __repr__(self):
        return f"face_explainer(mode={self.mode}, fraction={self.fraction}, n_neighbors={self.n_neighbors})"

    def fit(self, model: Model, data: Dataset):
        """
        Fit FACE with a dataset and model. Builds the kNN graph and the sparse adjacency matrix.
        """
        self.model = model
        self.dataset_ref = data
        
        X_all = np.array(data.data)
        y_all = np.array(data.target)

        self.transformed_feature_names = []
        self.transformed_feature_names.extend(data.continuous_features)
        
        for feat in data.categorical_features:
            categories = data.categorical_values.get(feat, [])
            encoded_names = [f"{feat}_{str(val)}" for val in categories]
            self.transformed_feature_names.extend(encoded_names)

        if len(self.transformed_feature_names) != X_all.shape[1]:
            print(f"[WARN] Feature name mismatch. Generated {len(self.transformed_feature_names)} names "
                    f"but data has {X_all.shape[1]} columns. Falling back to generic indices.")
            self.transformed_feature_names = [str(i) for i in range(X_all.shape[1])]
            
        # 2. Subsampling (Optimization) and setting data for graph
        if 0 < self.fraction < 1 and int(len(X_all) * self.fraction) < len(X_all):
            n = max(2, int(len(X_all) * self.fraction))  # at least 2 samples
            idx = np.random.choice(len(X_all), size=n, replace=False)
            self.X_graph = X_all[idx]
            self.y_graph = y_all[idx]
        else:
            self.X_graph = X_all
            self.y_graph = y_all

        # 3. Build kNN graph and Adjacency Matrix
        effective_neighbors = min(self.n_neighbors, len(self.X_graph))
        self.neigh = NearestNeighbors(n_neighbors=effective_neighbors)
        self.neigh.fit(self.X_graph)

        # Build sparse adjacency matrix for Dijkstra's
        distances, indices = self.neigh.kneighbors(self.X_graph)
        n_samples = len(self.X_graph)
        self.adjacency_matrix = lil_matrix((n_samples, n_samples))
        
        for i in range(n_samples):
            for j, neighbor_idx in enumerate(indices[i]):
                # Use the calculated distance as the edge weight
                distance = distances[i][j]
                self.adjacency_matrix[i, neighbor_idx] = distance
                # Ensure symmetric edges (important for manifold)
                self.adjacency_matrix[neighbor_idx, i] = distance 

        self.adjacency_matrix = self.adjacency_matrix.tocsr() # Convert to CSR for efficiency


    def explain(self, model: Model, data: Dataset, y_desired: Optional[int] = None) -> List[Counterfactual]:
        counterfactuals = []
        
        # Note: Input data uses the transformed names
        df = pd.DataFrame(data.data, columns=self.transformed_feature_names) 
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
        instance_df = pd.DataFrame(instance_ds.data, columns=self.transformed_feature_names)
        target = target_class or self.desired_class or 1
        return self._generate_cf(instance_df, model, target)

    def _generate_cf(self, instance_df: pd.DataFrame, model: Model, target_class: int) -> Optional[Counterfactual]:
        """
        Helper to generate a single counterfactual using the true FACE approach (shortest path on manifold).
        """
        if self.adjacency_matrix is None:
            raise RuntimeError("Explainer must be fitted before generating explanations.")
            
        try:
            instance = instance_df[self.transformed_feature_names].values[0]

            pred_probs = model.predict_proba(instance.reshape(1, -1))
            pred_orig = np.argmax(pred_probs)
            
            if pred_orig == target_class:
                return None

            # 1. Find the Nearest Node in the Training Graph (Source Node for Dijkstra)
            # This is necessary because the input 'instance' is usually not one of the graph nodes (self.X_graph)
            source_distance, source_index_list = self.neigh.kneighbors(instance.reshape(1, -1), n_neighbors=1)
            source_index = source_index_list[0][0]

            # 2. Find Shortest Path Distances to all other Nodes (Dijkstra's Algorithm)
            # shortest_path returns a matrix (1 x N_nodes) of distances from the source node
            geodesic_distances = shortest_path(
                csgraph=self.adjacency_matrix,
                directed=False,
                indices=source_index,
                method='D',
                return_predecessors=False
            )
            
            # The distances array returned by shortest_path is a 1-D array of geodesic distances from the source node
            # to every node in self.X_graph.
            
            # 3. Identify Target Nodes and Filter by Target Class
            geodesic_distances = geodesic_distances.flatten()
            target_mask = (self.y_graph == target_class)
            target_indices = np.where(target_mask)[0]
            
            if len(target_indices) == 0:
                return None

            # 4. Find the Closest Node in the Target Class on the Manifold
            min_geodesic_distance = np.inf
            best_candidate_index = -1
            
            for idx in target_indices:
                dist = geodesic_distances[idx]
                if dist < min_geodesic_distance:
                    min_geodesic_distance = dist
                    best_candidate_index = idx
                    
            if best_candidate_index == -1 or np.isinf(min_geodesic_distance):
                return None
            
            # 5. Select the Best Counterfactual
            candidate_cf = self.X_graph[best_candidate_index]

            return Counterfactual(
                original_data=instance,
                data=candidate_cf,
                original_class=pred_orig,
                target_class=target_class,
                explainer=repr(self),
            )

        except Exception as e:
            print(f"[WARN] FACE failed for instance: {e}")
            return None