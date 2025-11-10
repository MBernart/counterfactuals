import io
from typing import List, Tuple, Any, Dict, Union
import numpy as np
import pandas as pd
import pickle
from .counterfactual import ClassLabel
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class Dataset:
    """This is a helper class"""

    def __init__(
        self,
        df: pd.DataFrame,
        target: Union[List[ClassLabel], np.ndarray],
        features: Union[List[str], None] = None,
        immutable_features: List[str] = [],
        categorical_features: List[str] = [],
        categorical_values: Dict[str, List[Any]] = {},
        continuous_features: List[str] = [],
        allowable_ranges: Dict[str, Tuple[float, float]] = {},
        preprocessor: Union[ColumnTransformer, None] = None,
    ):
        self.df = df
        self.target: List[ClassLabel] = target.tolist() if isinstance(target, np.ndarray) else target

        self.features             = features
        self.immutable_features   = immutable_features

        self.categorical_features = categorical_features
        self.categorical_values   = categorical_values

        self.continuous_features  = continuous_features
        self.allowable_ranges     = allowable_ranges

        self._ensure_features()
        self._fill_categorical_values()
        self._fill_allowable_ranges()

        self.categorical_features_ids = [self.features.index(f) for f in self.categorical_features]
        self.continuous_features_ids  = [self.features.index(f) for f in self.continuous_features]
        self.immutable_features_ids   = [self.features.index(f) for f in self.immutable_features]

        if preprocessor is None:
            self.preprocessor = self.get_preprocessor()
            self.data: np.ndarray = self.preprocessor.fit_transform(self.df)
        else:
            self.preprocessor = preprocessor
            self.data: np.ndarray = self.preprocessor.transform(self.df)

    def _ensure_features(self):
        cat_features = set(self.categorical_features)
        num_features = set(self.continuous_features)
        immutable_features = set(self.immutable_features)
        all_processing_features = cat_features.union(num_features)
        overlapping_cat_num = cat_features.intersection(num_features)

        if self.features is None:
            all_features = all_processing_features
            self.features = list(all_features)
        else:
            all_features = set(self.features)

        assert all_features != set(), (
            "No features were defined. Please provide 'features' or "
            "at least one of 'categorical_features' or 'continuous_features'."
        )

        assert overlapping_cat_num == set(), (
            f"Features cannot be both categorical and continuous: {overlapping_cat_num}"
        )

        assert all_processing_features == all_features, (
            "The union of categorical and continuous features does not match the "
            f"complete feature list. Missing: {all_features - all_processing_features}. "
            f"Extra: {all_processing_features - all_features}"
        )

        assert immutable_features.issubset(all_features), (
            "Immutable features must be a subset of all features. "
            f"Unknown immutable features: {immutable_features - all_features}"
        )

    def _fill_categorical_values(self):
        cat_features_set = set(self.categorical_features)
        defined_cat_keys = set(self.categorical_values.keys())

        assert defined_cat_keys.issubset(cat_features_set), (
            f"Keys in 'categorical_values' must be a subset of 'categorical_features'. "
            f"Invalid keys found: {defined_cat_keys - cat_features_set}"
        )

        for feat in self.categorical_features:
            if feat not in self.categorical_values:
                self.categorical_values[feat] = np.unique(self.df[feat].values).tolist()

    def _fill_allowable_ranges(self):
        cont_features_set = set(self.continuous_features)
        defined_range_keys = set(self.allowable_ranges.keys())

        assert defined_range_keys.issubset(cont_features_set), (
            f"Keys in 'allowable_ranges' must be a subset of 'continuous_features'. "
            f"Invalid keys found: {defined_range_keys - cont_features_set}"
        )

        for feat in self.continuous_features:
            if feat not in self.allowable_ranges:
                values = self.df[feat].values
                self.allowable_ranges[feat] = (values.min(), values.max())

    def get_preprocessor(self) -> ColumnTransformer:
        """
        Creates a ColumnTransformer with pipelines for numerical and
        categorical features.
        """
        num_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categories = [self.categorical_values[feat] for feat in self.categorical_features]
        cat_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(categories=categories, handle_unknown='error', sparse_output=False))
        ])

        return ColumnTransformer(
            transformers=[
                ('num', num_transformer, self.continuous_features),
                ('cat', cat_transformer, self.categorical_features)
            ],
            remainder='drop'
        )

    def inverse_transform(self, data: np.ndarray) -> pd.DataFrame:
        """
        Inverse transforms preprocessed data back into a DataFrame
        in the original feature space.
        """
        num_features_len = len(self.continuous_features)

        data_num = data[:, :num_features_len]
        data_cat = data[:, num_features_len:]

        try:
            inv_data_num = self.preprocessor.named_transformers_['num'].inverse_transform(data_num)
            inv_data_cat = self.preprocessor.named_transformers_['cat'].inverse_transform(data_cat)
        except Exception as e:
            print(f"Error during inverse transform: {e}")
            print("Ensure the input data shape matches the preprocessor's output.")
            print(f"Expected num features: {num_features_len}, cat features: {len(self.categorical_features)}")
            print(f"Received data shape: {data.shape}")
            raise

        inv_data_full = np.hstack((inv_data_num, inv_data_cat))

        column_names = self.continuous_features + self.categorical_features
        df_reconstructed = pd.DataFrame(inv_data_full, columns=column_names)

        original_order = [f for f in self.features if f in df_reconstructed.columns]        
        return df_reconstructed[original_order]

    class DatasetIterator:
        def __init__(self, dataset: "Dataset"):
            self.dataset = dataset
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.index >= len(self.dataset.data):
                raise StopIteration
            result = self.dataset[self.index]
            self.index += 1
            return result

    def __iter__(self) -> DatasetIterator:
        return Dataset.DatasetIterator(self)

    def __getitem__(self, key) -> "Dataset":
        if isinstance(key, slice):
            data = self.data[key.start : key.stop : key.step]
            target = (
                self.target[key.start : key.stop : key.step]
                if self.target is not None
                else None
            )
        elif isinstance(key, int):
            data = self.data[key : key + 1]
            target = self.target[key : key + 1] if self.target is not None else None
        else:
            raise TypeError("Invalid argument type.")
        return self.like(data, target)

    def like(self, data: np.ndarray, target: np.ndarray) -> "Dataset":
        return Dataset(
            self.inverse_transform(data),
            target,
            self.features,
            immutable_features=self.immutable_features,
            categorical_features=self.categorical_features,
            categorical_values=self.categorical_values,
            continuous_features=self.continuous_features,
            allowable_ranges=self.allowable_ranges,
            preprocessor=self.preprocessor
        )
    
    @staticmethod
    def _dataframe_to_bytes(df: pd.DataFrame) -> bytes:
        """Helper to serialize a single pd.DataFrame to Parquet bytes."""
        with io.BytesIO() as f:
            df.to_parquet(f, engine='pyarrow')
            return f.getvalue()

    @staticmethod
    def _bytes_to_dataframe(b: bytes) -> pd.DataFrame:
        """Helper to deserialize Parquet bytes back into a single pd.DataFrame."""
        with io.BytesIO(b) as f:
            return pd.read_parquet(f, engine='pyarrow')

    def serialize(self) -> bytes:
        return pickle.dumps(
            {
                "df": Dataset._dataframe_to_bytes(self.df),
                "target": self.target,
                "features": self.features,
                "immutable_features": self.immutable_features,
                "categorical_features": self.categorical_features,
                "categorical_values": self.categorical_values,
                "continuous_features": self.continuous_features,
                "allowable_ranges": self.allowable_ranges,
                "preprocessor": self.preprocessor,
            },
            protocol=4
        )

    @staticmethod
    def deserialize(data: bytes) -> "Dataset":
        obj = pickle.loads(data)
        return Dataset(
            Dataset._bytes_to_dataframe(obj["df"]),
            obj["target"],
            obj["features"],
            immutable_features=obj["immutable_features"],
            categorical_features=obj["categorical_features"],
            categorical_values=obj["categorical_values"],
            continuous_features=obj["continuous_features"],
            allowable_ranges=obj["allowable_ranges"],
            preprocessor=obj["preprocessor"]
        )
