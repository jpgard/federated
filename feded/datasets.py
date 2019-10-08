"""
Functions to load and transform datasets.
"""
import collections
import functools

import pandas as pd
import tensorflow as tf
import numpy as np

from feded.preprocessing import convert_object_columns_to_numeric

DEFAULT_LARC_TARGET_COLNAME = "CRSE_GRD_OFFCL_CD"
DEFAULT_LARC_FEATURE_COLNAMES = [
    "CLASS_NBR",
    # "CRSE_GRD_OFFCL_CD",
    "EXCL_CLASS_CUM_GPA",
    # "SBJCT_CD",
    "GRD_BASIS_ENRL_CD",
    "TERM_CD",
]
NUM_EPOCHS = 10
BATCH_SIZE = 20
SHUFFLE_BUFFER = 50

CATEGORIES = {
    # "CRSE_GRD_OFFCL_CD": ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-",
    #                       "D+", "D", "D-", "E", "F"],
    "GRD_BASIS_ENRL_CD": ["GRD", "NON", "SUS", "OPF", "AUD"]
}

NUMERIC_FEATURES = ["CLASS_NBR",
                    "EXCL_CLASS_CUM_GPA",
                    "TERM_CD"]


def make_tff_data(df, target_colname=DEFAULT_LARC_TARGET_COLNAME,
                  feature_colnames=DEFAULT_LARC_FEATURE_COLNAMES):
    """
    Convert a pd.DataFrame into a list, where each element holds the data of an individual
        user as a tf.data.Dataset.
    """

    # TODO(jpgard): use repeat/shuffle/batch here; currently only generates a single
    #  iteration
    #  over dataset.
    def element_fn(element):
        x = element[feature_colnames].values
        y = element[target_colname]
        return collections.OrderedDict([
            ('x', tf.reshape(x, [-1])),
            ('y', tf.reshape(y, [1])),
        ])

    return [element_fn(element) for i, element in df.iterrows()]


def preprocess(dataset):
    # TODO(jpgard): build a proper feature vector from multiple input
    # features and with the correct target, transformed to numeric.

    # element_fn extracts feature and label vectors from each element;
    # 'x' and 'y' are used by keras.
    def element_fn(element):
        return collections.OrderedDict([
            ('x', tf.reshape(element['CLASS_NBR'], [-1])),
            ('y', tf.reshape(element['EXCL_CLASS_CUM_GPA'], [1])),
        ])
    return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
        SHUFFLE_BUFFER).batch(BATCH_SIZE)


def get_categorical_columns(categories=CATEGORIES):
    #TODO(jpgard): should this only contain input features?
    categorical_columns = []
    for feature, vocab in categories.items():
        cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
            key=feature, vocabulary_list=vocab)
        categorical_columns.append(tf.feature_column.indicator_column(cat_col))
    return categorical_columns


class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_freatures = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_freatures]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features

        return features, labels


def normalize_numeric_data(data, mean, std):
    # Center the data
    return (data - mean) / std


def get_numeric_columns(train_file_path):
    desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()
    MEAN = np.array(desc.T['mean'])
    STD = np.array(desc.T['std'])
    normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)
    numeric_column = tf.feature_column.numeric_column(
        'numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
    numeric_columns = [numeric_column]
    return numeric_columns


def make_larc_dataset(fp, target_colname=DEFAULT_LARC_TARGET_COLNAME,
                      feature_colnames=DEFAULT_LARC_FEATURE_COLNAMES):
    """
    Fetch LARC CSV from fp and load it as a tf.data.Dataset.
    :param fp:
    :return:
    """
    # TODO(jpgard): this is deprecated in favor of create_tf_dataset_for_client_fn()
    #  below. Remove it.
    colnames_to_keep = feature_colnames + [target_colname]
    # df = pd.read_csv(fp, usecols=colnames_to_keep)
    # # apply transformations to convert any object columns to
    # #  discrete numeric values via pd.Categorical();
    # #  see https://www.tensorflow.org/tutorials/load_data/pandas_dataframe.
    # df = convert_object_columns_to_numeric(df)
    # # target = df.pop(target_colname)
    # dataset = tf.data.Dataset.from_tensor_slices(df.values)
    dataset = tf.data.experimental.make_csv_dataset(
        fp, batch_size=BATCH_SIZE, select_columns=colnames_to_keep,
        num_epochs=NUM_EPOCHS, shuffle=True, shuffle_buffer_size=SHUFFLE_BUFFER,
        label_name=target_colname
    )

    packed_numeric_dataset = dataset.map(PackNumericFeatures(NUMERIC_FEATURES))

    # return make_tff_data(df, target_colname=target_colname,
    #                      feature_colnames=feature_colnames)

    return packed_numeric_dataset


def create_larc_tf_dataset_for_client(client_id, fp,
                                      target_colname=DEFAULT_LARC_TARGET_COLNAME,
                                      feature_colnames=DEFAULT_LARC_FEATURE_COLNAMES):
    """
    A function that takes a client_id from the above list, and returns a tf.data.Dataset.
    See #https://www.tensorflow.org/federated/api_docs/python/tff/simulation/ClientData
    :param client_id:
    :return:
    """
    # TODO(jpgard): take action based on client id; currently just returns the entire
    #  dataset contained in the csv.
    colnames_to_keep = feature_colnames + [target_colname]
    dataset = tf.data.experimental.make_csv_dataset(
        fp,
        batch_size=1,
        select_columns=colnames_to_keep,
        # num_epochs=NUM_EPOCHS,
        shuffle=True,
        shuffle_buffer_size=SHUFFLE_BUFFER,
        # enabling this is primarily useful for tf.Estimator API.
        # label_name=target_colname
    )
    return dataset