import io
from typing import Optional, List, Tuple, Any, Dict
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
        target: List[ClassLabel] | np.ndarray,
        features: List[str],
        immutable_features: List[str] = [],
        categorical_features: List[str] = [],
        categorical_values: Dict[str, List[Any]] = {},
        continuous_features: List[str] = [],
        allowable_ranges: Dict[str, Tuple[float, float]] = {},
    ):
        self.df = df
        self.target: List[ClassLabel] = target.tolist() if isinstance(target, np.ndarray) else target

        self.features             = features
        self.immutable_features   = immutable_features

        self.categorical_features = categorical_features
        self.categorical_values   = categorical_values

        self.continuous_features  = continuous_features
        self.allowable_ranges     = allowable_ranges

        self.categorical_features_ids = [features.index(f) for f in categorical_features]
        self.continuous_features_ids  = [features.index(f) for f in continuous_features]
        self.immutable_features_ids   = [features.index(f) for f in immutable_features]

        self.preprocessor = self.get_preprocessor()
        self.data: np.ndarray = self.preprocessor.fit_transform(self.df)

    def get_preprocessor(self) -> ColumnTransformer:
        """
        Creates a ColumnTransformer with pipelines for numerical and
        categorical features.
        """
        self._numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        self._categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='error', sparse_output=False))
        ])

        return ColumnTransformer(
            transformers=[
                ('num', self._numerical_transformer, self.continuous_features),
                ('cat', self._categorical_transformer, self.categorical_features)
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

    def like(self, data: np.ndarray, target: Optional[np.ndarray] = None) -> "Dataset":
        if target is None:
            target = self.target
        # TODO: refactor __class__ call
        return self.__class__(
            data,
            target,
            self.features,
            self.categorical_features,
            self.continuous_features,
            self.immutable_features,
            self.allowable_ranges,
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
        )
