"""
Train a FedEd model.
"""
import argparse
import six
import pandas as pd
import tensorflow as tf

from functools import partial

import tensorflow_federated as tff

NUM_CLIENTS = 3
EPOCHS = 5

# NOTE: If the statement below fails, it means that you are
# using an older version of TFF without the high-performance
# executor stack. Call `tff.framework.set_default_executor()`
# instead to use the default reference runtime.
if six.PY3:
  tff.framework.set_default_executor(
      tff.framework.create_local_executor(NUM_CLIENTS))

from feded.datasets import preprocess, create_larc_tf_dataset_for_client, NUM_EPOCHS, \
    DEFAULT_LARC_CLIENT_COLNAME
from feded.model import create_compiled_keras_model


def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x))
            for x in client_ids]


def main(data_fp):
    # # fetch and preprocess the data
    create_tf_dataset_for_client_fn = partial(create_larc_tf_dataset_for_client,
                                              fp=data_fp)
    # TODO(jpgard): move this into a function that fetches client ids; optionally
    #  should apply some sort of threshold t, only returning data from clients with
    #  count(client) >= t
    # client_ids = pd.read_csv(data_fp,
    #     usecols=[DEFAULT_LARC_CLIENT_COLNAME])[
    #     #     DEFAULT_LARC_CLIENT_COLNAME].unique()
    client_ids = ["ECON", "MATH", "ENGLISH", "MATSCIE"]
    feded_train = tff.simulation.ClientData.from_clients_and_fn(
        client_ids=client_ids,
        create_tf_dataset_for_client_fn=create_tf_dataset_for_client_fn
    )
    feded_test = tff.simulation.ClientData.from_clients_and_fn(
        client_ids=client_ids,
        create_tf_dataset_for_client_fn=create_tf_dataset_for_client_fn
    )

    example_dataset = feded_train.create_tf_dataset_for_client(
        feded_train.client_ids[0]
    )

    # example_element = iter(example_dataset).next()
    # print(example_element)
    preprocessed_example_dataset = preprocess(example_dataset)

    sample_batch = tf.nest.map_structure(
        lambda x: x.numpy(), iter(preprocessed_example_dataset).next())

    def model_fn():
        keras_model = create_compiled_keras_model(input_shape=(sample_batch['x'].shape[1],))
        return tff.learning.from_compiled_keras_model(keras_model,
                                                      sample_batch)

    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn)
    print(iterative_process.initialize.type_signature)
    state = iterative_process.initialize()

    # fetch the federated training data and execute an iteration of training
    train_client_ids = feded_train.client_ids[:NUM_CLIENTS]
    federated_train_data = make_federated_data(feded_train, train_client_ids)

    for i in range(NUM_EPOCHS):
        state, metrics = iterative_process.next(state, federated_train_data)
        print('round  {}, metrics={}'.format(i, metrics))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fp", help="path to data csv")
    args = parser.parse_args()
    main(**vars(args))
