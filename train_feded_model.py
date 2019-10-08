"""
Train a FedEd model.
"""
import argparse
import tensorflow as tf
from functools import partial

tf.compat.v1.enable_v2_behavior()

import tensorflow_federated as tff
from feded.datasets import make_larc_dataset, preprocess, get_categorical_columns, \
    get_numeric_columns, create_larc_tf_dataset_for_client
from feded.model import create_compiled_keras_model


def show_batch(dataset):
    # a utility function for debugging
    # via https://www.tensorflow.org/tutorials/load_data/csv
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))


def pack(features, label):
    # pack together all columns into a single feature vector
    # https://www.tensorflow.org/tutorials/load_data/csv
    return tf.stack(list(features.values()), axis=-1), label


def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x))
            for x in client_ids]


def main(data_fp):
    # # fetch and preprocess the data
    # example_dataset = make_larc_dataset(data_fp)
    # example_batch, labels_batch = next(iter(example_dataset))
    # sample_batch = tf.nest.map_structure(
    #     lambda x: x.numpy(), iter(example_dataset).next())
    # # TODO(jpgard): use preprocessing steps in
    # # https://www.tensorflow.org/tutorials/load_data/csv to transform the data;
    # # specifically we need to convert any categorical columns to int/float
    # # and get all data of same type before applying pack().
    # categorical_columns = get_categorical_columns()
    # numeric_columns = get_numeric_columns(data_fp)
    #
    # # test the numeric and categorical columns
    # numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
    # print(numeric_layer(example_batch).numpy())
    # categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)
    # print(categorical_layer(example_batch).numpy()[0])
    #
    # preprocessing_layer = tf.keras.layers.DenseFeatures(
    #     categorical_columns + numeric_columns)
    #
    # print(preprocessing_layer(example_batch).numpy()[0])

    create_tf_dataset_for_client_fn = partial(create_larc_tf_dataset_for_client,
                                              fp=data_fp)
    feded_train = tff.simulation.ClientData.from_clients_and_fn(
        client_ids=["1", "2", "3"],
        create_tf_dataset_for_client_fn=create_tf_dataset_for_client_fn
    )
    feded_test = tff.simulation.ClientData.from_clients_and_fn(
        client_ids=["1", "2", "3"],
        create_tf_dataset_for_client_fn=create_tf_dataset_for_client_fn
    )

    example_dataset = feded_train.create_tf_dataset_for_client(
        feded_train.client_ids[0])

    example_element = iter(example_dataset).next()
    print(example_element)
    preprocessed_example_dataset = preprocess(example_dataset)
    sample_batch = tf.nest.map_structure(
        lambda x: x.numpy(), iter(preprocessed_example_dataset).next())

    # TODO(jpgard): we should now have an acceptable dataset format. Complete a full
    # example pipeline of training from here; will need to remove the preprocessing
    # layers (and any other hacks added during development) and use a set of vanilla
    # fully-connected layers. Later, add things like preprocessing or normalization as
    # required/appropriate.

    def model_fn():
        keras_model = create_compiled_keras_model() #preprocessing_layer
        return tff.learning.from_compiled_keras_model(keras_model,
                                                      sample_batch)

    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn)
    print(iterative_process.initialize.type_signature)
    state = iterative_process.initialize()

    # fetch the federated training data and execute an iteration of training
    train_client_ids = feded_train.client_ids[:]
    federated_train_data = make_federated_data(feded_train, train_client_ids)
    state, metrics = iterative_process.next(state, federated_train_data)
    print('round  1, metrics={}'.format(metrics))
    import ipdb;ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fp", help="path to data csv")
    args = parser.parse_args()
    main(**vars(args))
