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
        feature_values_dict[f] = df[feature_values_dict].unique()
    return feature_values_dict
