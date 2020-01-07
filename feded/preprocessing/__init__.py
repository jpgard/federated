"""
Functions for preprocessing data.
"""

import collections
import glob
from pandas.api.types import is_numeric_dtype, is_object_dtype
import pandas as pd
import tensorflow as tf



def preprocess(dataset: tf.data.Dataset, feature_layer: tf.keras.layers,
               target_feature: str,
               num_epochs: int,
               shuffle_buffer: int,
               batch_size: int,
               batches_to_take=None):
    """
    Preprocess data with a single-element label (label of length one).

    :param dataset: the dataset to preprocess.
    :param feature_layer: feature layer to use to preprocess the data.
    :param target_feature: the name of the target feature (used to extract the correct
    element from the input observations).
    :param num_epochs: number of epochs to repeat for; by default, it is set to.
    :return:
    """

    def element_fn(element):
        # element_fn extracts feature and label vectors from each element;
        # 'x' and 'y' names are required by keras.
        feature_vector = feature_layer(element)

        return collections.OrderedDict([
            ('x', tf.reshape(feature_vector, [feature_vector.shape[1]])),
            ('y', tf.reshape(element[target_feature], [1])),
        ])

    preprocessed_dataset = dataset.repeat(num_epochs).map(element_fn).shuffle(
        shuffle_buffer).batch(batch_size)
    if not batches_to_take:
        return preprocessed_dataset
    else:
        return preprocessed_dataset.take(batches_to_take)


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
    binary_indicator = df.loc[:, colname].isin(positive_vals).astype(int)
    if replace:
        df.loc[:, colname] = binary_indicator
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


def minmax_scale_numeric_columns(df: pd.DataFrame, columns: list):
    """
    Scale numeric columns to the range (0,1) using (min_value, max_value).
    :param df:
    :param columns:
    :return: df with the original columns modified.
    """
    for column in columns:
        assert is_numeric_dtype(df[column]), "only pass numeric columns to minmax scaler"
        col_max = df[column].max()
        col_min = df[column].min()
        if col_max == col_min:  # case: no scaling needed; just center
            df.loc[:, column] = df[column] - col_min
        else:  # case: normal case; center and scale
            df.loc[:, column] = (df[column] - col_min) / col_max
    return df


def generate_missing_value_indicator(df: pd.DataFrame, columns: list, fill_value="NA"):
    """Fill any na values in columns with fill_value."""
    for column in columns:
        if not is_object_dtype(df[column]):
            print("skipping non-object column {}".format(column))
        else:
            df.loc[:, column] = df[column].fillna(fill_value)
    return df
