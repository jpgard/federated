"""
Functions for preprocessing data.
"""


import collections

import pandas as pd
import tensorflow as tf

from feded.config import TrainingConfig
from feded.datasets.larc import DEFAULT_LARC_TARGET_COLNAME

import glob
import pandas as pd


def preprocess(dataset, feature_layer, training_config: TrainingConfig,
               target_feature=DEFAULT_LARC_TARGET_COLNAME):
    """Preprocess data with a single-element label (label of length one)."""
    num_epochs = training_config.epochs
    shuffle_buffer = training_config.shuffle_buffer
    batch_size = training_config.batch_size

    def element_fn(element):
        # element_fn extracts feature and label vectors from each element;
        # 'x' and 'y' names are required by keras.
        feature_vector = feature_layer(element)

        return collections.OrderedDict([
            ('x', tf.reshape(feature_vector, [feature_vector.shape[1]])),
            ('y', tf.reshape(element[target_feature], [1])),
        ])

    return dataset.repeat(num_epochs).map(element_fn).shuffle(shuffle_buffer).batch(
        batch_size)


def generate_categorical_feature_dict(df: pd.DataFrame, categorical_features: list):
    """
    Create a dictionary of {feature_name: [unique_values,]}.
    :param df: pd.DataFrame containing categorical_features as names.
    :param categorical_features: list of column names to process.
    :return: Dictionary with format specified above.
    """
    feature_values_dict = dict()
    for f in categorical_features:
        feature_values_dict[f] = df[f].unique()
    return feature_values_dict


def filter_df_by_values(df: pd.DataFrame, colname: str, keep_vals: list):
    """Filter the rows of df to keep only those rows containing valkues in keep_vals."""
    return df[df[colname].isin(keep_vals)]


def make_binary_indicator_column(df: pd.DataFrame, colname: str, positive_vals: list,
                                 replace=False, newname=None):
    """Create a binary indicator column where rows with values for colname which are in
        positive_vals take a value of 1 and all others take 0."""
    binary_indicator = df[colname].isin(positive_vals).astype(int)
    if replace:
        df[colname] = binary_indicator
    else:
        assert newname, "must provide a new name if not replacing existing column."
        df[newname] = binary_indicator
    return df


def read_csv(fp_or_wildcard: str, **kwargs):
    """
    Provides a pd.read_csv() -style interface supporting use of wildcards.
    :param fp_or_wildcard: filepath or wildcard to read from.
    :param kwargs: arguments passed to read.csv
    :return:
    """
    data = list()
    for fp in glob.glob(fp_or_wildcard):
        print("[INFO] reading data from {}".format(fp))
        data.append(pd.read_csv(fp, **kwargs))
    return pd.concat(data, axis=0, ignore_index=True)
