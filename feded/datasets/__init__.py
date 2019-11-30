"""
Functions and classes to represent datasets.
"""

from abc import ABC, abstractmethod
from typing import Optional, List

from feded.config import TrainingConfig


class TabularDataset(ABC):
    """A class to represent tabular datasets."""

    def __init__(self, client_id_col: str,
                 target_column: str,
                 categorical_columns: Optional[List[str]] = None,
                 embedding_columns: Optional[List[str]] = None,
                 numeric_columns: Optional[List[str]] = None,
                 ):
        self.client_id_col = client_id_col
        self.categorical_columns = categorical_columns
        self.df = None
        self.embedding_columns = embedding_columns
        self.numeric_columns = numeric_columns
        self.target_column = target_column

    @property
    def feature_column_names(self):
        """The name of all feature columns; excluding the client_id column."""
        return self.categorical_columns + self.embedding_columns + self.numeric_columns

    @abstractmethod
    def read_data(self, fp):
        raise  # this line should never be reached

    @abstractmethod
    def create_tf_dataset_for_client(self, client_id, training_config: TrainingConfig):
        """"Creates a tf.Dataset for the given client id."""
        raise  # this line should never be reached

    @abstractmethod
    def make_feature_layer(self):
        """Create a keras.DenseFeatures layer."""
        raise

    @property
    def client_ids(self):
        """The ids of all clients."""
        return self.df[self.client_id_col].unique().tolist()