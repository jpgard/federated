"""
Functions to load and transform datasets.
"""
import pandas as pd
import tensorflow as tf
from feded.preprocessing import convert_object_columns_to_numeric

DEFAULT_LARC_TARGET_COLNAME = "CRSE_GRD_OFFCL_CD"
DEFAULT_LARC_COLNAMES_TO_KEEP = [
    "CLASS_NBR",
    "CRSE_GRD_OFFCL_CD",
    "EXCL_CLASS_CUM_GPA",
    "SBJCT_CD",
    "TERM_CD",
]


def fetch_larc_dataset(fp, target_colname=DEFAULT_LARC_TARGET_COLNAME,
                       colnames_to_keep=DEFAULT_LARC_COLNAMES_TO_KEEP):
    """
    Fetch LARC CSV from fp and load it as a tf.data.Dataset.
    :param fp:
    :return:
    """
    df = pd.read_csv(fp, usecols=colnames_to_keep)
    # apply transofrmations to convert any object columns to
    #  discrete numeric values via pd.Categorical();
    #  see https://www.tensorflow.org/tutorials/load_data/pandas_dataframe.
    df = convert_object_columns_to_numeric(df)
    target = df.pop(target_colname)
    return tf.data.Dataset.from_tensor_slices((df.values, target.values))

