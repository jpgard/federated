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
    "CRSE_GRD_OFFCL_CD": ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-",
                          "D+", "D", "D-", "E", "F"],
    "GRD_BASIS_ENRL_CD": ["GRD", "NON", "SUS", "OPF", "AUD"]
}

NUMERIC_FEATURES = ["CATLG_NBR",
                    "CLASS_NBR",
                    "EXCL_CLASS_CUM_GPA",
                    "TERM_CD"]

CATEGORICAL_FEATURES = [
    "CRSE_GRD_OFFCL_CD",
    "GRD_BASIS_ENRL_CD"
]

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


def make_feature_layer():
    """
    Utility function to assemble a feature layer; this builds a single dense feature
    from a list of feature columns.
    :return: a tf.keras.layers.DenseFeatures layer.
    """
    fco_catlg = tf.feature_column.numeric_column("CATLG_NBR")
    fco_class = tf.feature_column.numeric_column("CLASS_NBR")
    fco_term = tf.feature_column.numeric_column("TERM_CD")
    fco = tf.feature_column.categorical_column_with_vocabulary_list(
        "CRSE_GRD_OFFCL_CD", CATEGORIES["CRSE_GRD_OFFCL_CD"])
    # one-hot encode the FeatureColumn and then compute it as a dense feature
    fco_ohe = tf.feature_column.indicator_column(fco)
    feature_columns = [fco_catlg, fco_class, fco_term, fco_ohe]
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    return feature_layer


def preprocess(dataset):

    feature_layer = make_feature_layer()

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


def create_larc_tf_dataset_for_client(client_id, fp, client_id_col=None,
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
    # use of tf.data.experimental.make_csv_dataset; this should only be used when the
    # entire CSV file represents a single clients' data).

    # dataset = tf.data.experimental.make_csv_dataset(
    #     fp,
    #     batch_size=1,
    #     num_epochs=NUM_EPOCHS,
    #     select_columns=colnames_to_keep,
    #     shuffle=True,
    #     shuffle_buffer_size=SHUFFLE_BUFFER,
    #     na_value=''
    # )


    df = pd.read_csv(fp, usecols=colnames_to_keep, na_values=('', ' '),
                     keep_default_na=True)
    # TODO(jpgard): handle NA values instead of dropping incomplete cases; for some
    # features (e.g. categorical features) we should generate an indicator for
    # missingness; for missing numeric features these will likely need to be dropped.
    df.dropna(inplace=True)
    dataset = tf.data.Dataset.from_tensor_slices(df.to_dict('list'))
    dataset = dataset.shuffle(SHUFFLE_BUFFER).batch(1).repeat(NUM_EPOCHS)
    return dataset