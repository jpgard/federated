"""
Functions for preprocessing data.
"""
import pandas as pd

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
                                 replace=False, newname = None):
    """Create a binary indicator column where rows with values for colname which are in
        positive_vals take a value of 1 and all others take 0."""
    binary_indicator = df[colname].isin(positive_vals).astype(int)
    if replace:
        df[colname] = binary_indicator
    else:
        assert newname, "must provide a new name if not replacing existing column."
        df[newname] = binary_indicator
    return df