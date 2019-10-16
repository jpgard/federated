"""
Functions to load and transform datasets.
"""
import collections
import functools

import pandas as pd
import tensorflow as tf
import numpy as np

DEFAULT_LARC_TARGET_COLNAME = "CRSE_GRD_OFFCL_CD"
DEFAULT_LARC_FEATURE_COLNAMES = [
    "CATLG_NBR",
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

NUMERIC_FEATURES = ["CATLG_NBR",
                    "CLASS_NBR",
                    "EXCL_CLASS_CUM_GPA",
                    "TERM_CD"]


def preprocess(dataset):
    # TODO(jpgard): build a proper feature vector from multiple input
    # features and with the correct target, transformed to numeric.

    # element_fn extracts feature and label vectors from each element;
    # 'x' and 'y' are used by keras.
    def element_fn(element):
        return collections.OrderedDict([
            ('x', tf.reshape(element['CATLG_NBR'], [1])),
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


def create_larc_tf_dataset_for_client(client_id, fp,
                                      target_colname=DEFAULT_LARC_TARGET_COLNAME,
                                      feature_colnames=DEFAULT_LARC_FEATURE_COLNAMES,
                                      ):
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
        num_epochs=NUM_EPOCHS,
        select_columns=colnames_to_keep,
        shuffle=True,
        shuffle_buffer_size=SHUFFLE_BUFFER,
        # enabling this is primarily useful for tf.Estimator API.
        # label_name=target_colname
    )
    # TODO(jpgard): possibly uncomment below
    # return dataset.map(PackNumericFeatures(NUMERIC_FEATURES))
    return dataset