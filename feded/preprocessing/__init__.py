import collections

import tensorflow as tf

from feded.config import TrainingConfig
from feded.datasets.larc import DEFAULT_LARC_TARGET_COLNAME


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