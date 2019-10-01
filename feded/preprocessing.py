"""
Functions for preprocessing data.
"""
import pandas as pd

def convert_object_columns_to_numeric(df):
    """
    Convert any object columns in df to discrete numeric columns.
    :param df: pd.DataFrame to convert.
    :return: pd.DataFrame
    """
    for colname,coltype in df.dtypes.iteritems():
        if coltype == "object":
            df[colname] = pd.Categorical(df[colname])
            df[colname] = df[colname].cat.codes
    return df
