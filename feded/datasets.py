"""
Functions to load and transform datasets.
"""
import collections
import functools

import pandas as pd
import tensorflow as tf
import numpy as np

from abc import ABC, abstractmethod
from feded.preprocessing import generate_categorical_feature_dict
from typing import Optional, Tuple, List

# the prediction target
DEFAULT_LARC_TARGET_COLNAME = "CRSE_GRD_OFFCL_CD"

# the column to use for generating clients; this should be a column which universities
# might use to partition their data (e.g. sharing across schools or departments might
# be restricted, so this would be a good choice).
DEFAULT_LARC_CLIENT_COLNAME = "SBJCT_CD"

# explicitly specify the categorical features and the values they can take
# TODO(jpgard): create a function which reads these directly from the data for a list
#  of specified categorical feature names; or even better, implement these as part of
#  an object representing the dataset.
DEFAULT_LARC_CATEGORICAL_FEATURES = [
    "CRSE_GRD_OFFCL_CD",
    "GRD_BASIS_ENRL_CD"

]

# CATEGORICAL_FEATURE_VALUES = {
#     "CRSE_GRD_OFFCL_CD": ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-",
#                           "D+", "D", "D-", "E", "F"],
#     "GRD_BASIS_ENRL_CD": ["GRD", "NON", "SUS", "OPF", "AUD"]
# }

# explicitly specify the numeric features
# TODO(jpgard): create a function which reads these directly from the data for a list
#  of specified feature names; or even better, implement these as part of an object
#  representing the dataset.
# TODO(jpgard): several of these should be categorical or ordinal features,
#  NOT numeric, as they are really just numeric identifiers for categorical features.
DEFAULT_LARC_NUMERIC_FEATURES = ["CATLG_NBR",
                                 "CLASS_NBR",
                                 "EXCL_CLASS_CUM_GPA",
                                 "TERM_CD"]

DEFAULT_LARC_EMBEDDING_FEATURES = []

# default training parameters
# TODO(jpgard): increase these later
BATCH_SIZE = 20
NUM_EPOCHS = 50
SHUFFLE_BUFFER = 10
BATCH_SIZE = 8


def preprocess(dataset, feature_layer):

    def element_fn(element):
        # element_fn extracts feature and label vectors from each element;
        # 'x' and 'y' names are required by keras.

        feature_vector = feature_layer(element)

        return collections.OrderedDict([
            ('x', tf.reshape(feature_vector, [feature_vector.shape[1]])),
            ('y', tf.reshape(element['EXCL_CLASS_CUM_GPA'], [1])),
        ])

    return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
        SHUFFLE_BUFFER).batch(BATCH_SIZE)


class TabularDataset(ABC):
    """A class to represent tabular datasets."""

    def __init__(self, client_id_col: str,
                 num_epochs: int = NUM_EPOCHS,
                 shuffle_buffer: int = SHUFFLE_BUFFER,
                 categorical_columns: Optional[List[str]] = None,
                 embedding_columns: Optional[List[str]] = None,
                 numeric_columns: Optional[List[str]] = None,
                 ):
        self.client_id_col = client_id_col
        self.categorical_columns = categorical_columns
        self.embedding_columns = embedding_columns
        self.numeric_columns = numeric_columns
        self.df = None
        self.num_epochs = num_epochs
        self.shuffle_buffer = shuffle_buffer

    @property
    def feature_column_names(self):
        """The name of all feature columns; excluding the client_id column."""
        return self.categorical_columns + self.embedding_columns + self.numeric_columns

    @abstractmethod
    def read_data(self, fp):
        raise  # this line should never be reached

    @abstractmethod
    def create_tf_dataset_for_client(self, client_id):
        """"Creates a tf.Dataset for the given client id."""
        raise  # this line should never be reached

    @abstractmethod
    def make_feature_layer(self):
        """Create a keras.DenseFeatures layer."""
        raise


class LarcDataset(TabularDataset):
    """A class to represent the LARC dataset."""

    def __init__(self, client_id_col: str = DEFAULT_LARC_CLIENT_COLNAME,
                 num_epochs: int = NUM_EPOCHS,
                 shuffle_buffer: int = SHUFFLE_BUFFER,
                 categorical_columns: List[str] = DEFAULT_LARC_CATEGORICAL_FEATURES,
                 embedding_columns: List[str] = DEFAULT_LARC_EMBEDDING_FEATURES,
                 numeric_columns: List[str] = DEFAULT_LARC_NUMERIC_FEATURES,
                 ):
        super(LarcDataset, self).__init__(client_id_col=client_id_col,
                                          categorical_columns=categorical_columns,
                                          embedding_columns=embedding_columns,
                                          numeric_columns=numeric_columns,
                                          num_epochs=num_epochs,
                                          shuffle_buffer=shuffle_buffer
                                          )

    def read_data(self, fp):
        colnames_to_keep = [self.client_id_col] + self.feature_column_names
        self.df = pd.read_csv(fp, usecols=colnames_to_keep, na_values=('', ' '),
                              keep_default_na=True)
        return

    def create_tf_dataset_for_client(self, client_id):
        # filter the dataset by client_id, keeping only the feature and target colnames
        df = self.df[self.df[self.client_id_col] == client_id][self.feature_column_names]
        if len(df):
            # TODO(jpgard): handle NA values instead of dropping incomplete cases; for
            #  some
            # features (e.g. categorical features) we can generate an indicator for
            # missingness; for missing numeric features these will likely need to be
            # dropped.
            df.dropna(inplace=True)
            dataset = tf.data.Dataset.from_tensor_slices(df.to_dict('list'))
            dataset = dataset.shuffle(self.shuffle_buffer).batch(1).repeat(
                self.num_epochs)
            return dataset
        else:
            print("[WARNING] no data for specified client_id {}".format(client_id))
            return None

    def make_feature_layer(self):
        """
        Utility function to assemble a feature layer; this builds a single dense feature
        from a list of feature columns.
        :return: a tf.keras.layers.DenseFeatures layer.
        """
        categorical_feature_values = generate_categorical_feature_dict(
            self.df, self.categorical_columns)
        # generate a list of feature columns
        feature_columns = list()
        for nc in self.numeric_columns:
            # TODO(jpgard): normalize the numeric columns
            feature_columns.append(tf.feature_column.numeric_column(nc))
        for cc in self.categorical_columns:
            # create one-hot encoded columns for each categorical column
            fco = tf.feature_column.categorical_column_with_vocabulary_list(
                    cc, categorical_feature_values[cc])
            fco_ohe = tf.feature_column.indicator_column(fco)
            feature_columns.append(fco_ohe)
        # TODO(jpgard): embedding columns here
        feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
        return feature_layer
