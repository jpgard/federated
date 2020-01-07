"""
Functions and classes to represent datasets.
"""

from abc import ABC, abstractmethod
from typing import Optional, List

from feded.training import TrainingConfig
import tensorflow as tf
import pandas as pd


def get_dataset_size(dataset: tf.data.Dataset):
    """Fetch the number of observations in a dataset."""
    return tf.data.experimental.cardinality(dataset).numpy()

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

    def filter(self, filter_spec: dict, in_place=False):
        """
        Filter self.df according to filter_spec.

        :param filter_spec: a dictionary with format {colname: [valid_values, ...]}
        :param in_place: if true, set the result to self.df; if false, return a copy
            of self.df with the filtering applied.
        :return: pd.DataFrame (if in_place=False), else None and self.df is modified.
        """
        def _filter_op(df, col, vals):
            return df[df[col].isin(vals)]
        if not in_place:
            df = self.df.copy(deep=True)
        for colname, valid_values in filter_spec.items():
            if not in_place:
                df = _filter_op(df, colname, valid_values)
            else:
                self.df = _filter_op(self.df, colname, valid_values)

    def from_df(self, df: pd.DataFrame):
        self.df = df


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
    def create_tf_dataset(self,  training_config: TrainingConfig):
        """"Create a single tf.Dataset to represent the entire (non-federated) dataset."""
        raise  # this line should never be reached

    @abstractmethod
    def make_feature_layer(self):
        """Create a keras.DenseFeatures layer."""
        raise

    @property
    def client_ids(self):
        """The ids of all clients."""
        return self.df[self.client_id_col].unique().tolist()
