"""
Train a FedEd model.

Usage (note the use of quites surrounding the wildcard path):

python train_feded_model.py \
    --data_fp "./data/larc-split/train/train/train_4*.csv" \
    --epochs 20 \
    --batch_size 512 \
    --shuffle_buffer 500 \
    --batches_to_take 64 \
    --num_train_clients 16

"""
import argparse
import six
import tensorflow as tf

import tensorflow_federated as tff

# NOTE: If the statement below fails, it means that you are
# using an older version of TFF without the high-performance
# executor stack. Call `tff.framework.set_default_executor()`
# instead to use the default reference runtime.
if six.PY3:
    tff.framework.set_default_executor(
        tff.framework.create_local_executor())

from feded.preprocessing import preprocess
from feded.datasets.larc import LarcDataset, DEFAULT_LARC_TARGET_COLNAME
from feded.model import create_compiled_keras_model
from feded.config import TrainingConfig
from feded.federated import sample_client_ids
from functools import partial


def make_federated_data(client_data, client_ids, feature_layer, training_config):
    """
    Apply preprocess to each client_id, but take only a single batch of training data.
    """
    preprocessing_fn = partial(preprocess,
                               feature_layer=feature_layer,
                               target_feature=DEFAULT_LARC_TARGET_COLNAME,
                               num_epochs=1,
                               shuffle_buffer=training_config.shuffle_buffer,
                               batch_size=training_config.batch_size,
                               batches_to_take=training_config.batches_to_take)
    return [preprocessing_fn(client_data.create_tf_dataset_for_client(x))
            for x in client_ids]


def make_sample_batch(dataset, feature_layer):
    """Make a sample batch from a randomly-selected client id."""
    example_dataset = dataset.create_tf_dataset_for_client(
        dataset.client_ids[0]
    )
    # preprocess the dataset
    preprocessed_example_dataset = preprocess(
        example_dataset, feature_layer,
        DEFAULT_LARC_TARGET_COLNAME,
        num_epochs=training_config.epochs,
        shuffle_buffer=training_config.shuffle_buffer,
        batch_size=training_config.batch_size
    )
    # fetch a single sample batch from the preprocessed dataset
    sample_batch = tf.nest.map_structure(
        lambda x: x.numpy(), iter(preprocessed_example_dataset).next())
    return sample_batch



def main(data_fp: str, training_config: TrainingConfig):
    # # fetch and preprocess the data
    dataset = LarcDataset()
    dataset.read_data(data_fp)
    create_tf_dataset_for_client_fn = lambda x: dataset.create_tf_dataset_for_client(
        x, training_config=training_config)
    feature_layer = dataset.make_feature_layer()

    client_ids = dataset.client_ids
    feded_train = tff.simulation.ClientData.from_clients_and_fn(
        client_ids=client_ids,
        create_tf_dataset_for_client_fn=create_tf_dataset_for_client_fn
    )
    feded_test = tff.simulation.ClientData.from_clients_and_fn(
        client_ids=client_ids,
        create_tf_dataset_for_client_fn=create_tf_dataset_for_client_fn
    )

    sample_batch = make_sample_batch(feded_train, feature_layer)

    def model_fn():
        keras_model = create_compiled_keras_model(
            input_shape=(sample_batch['x'].shape[1],))
        return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn)
    print(iterative_process.initialize.type_signature)
    state = iterative_process.initialize()

    # fetch the federated training data and execute an iteration of training
    train_client_ids = feded_train.client_ids
    # evaluation = tff.learning.build_federated_evaluation(model_fn)

    for i in range(training_config.epochs):
        client_ids = sample_client_ids(train_client_ids,
                                       training_config.num_train_clients, method="random")
        epoch_federated_train_data = make_federated_data(feded_train, client_ids,
                                                         feature_layer, training_config)
        # epoch_federated_test_data = make_federated_data(feded_test,
        #                                                 feded_test.client_ids,
        #                                                 feature_layer, training_config)
        state, train_metrics = iterative_process.next(state, epoch_federated_train_data)
        # test_metrics = evaluation(state.model, epoch_federated_test_data)
        print('round  {}, train_metrics={}'.format(i, train_metrics))
        # print('round  {}, test_metrics={}'.format(i, test_metrics))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fp", help="path to data csv")
    parser.add_argument("--epochs", type=int, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, help="batch size for training")
    parser.add_argument("--shuffle_buffer", type=int, help="shuffle buffer", default=500)
    parser.add_argument("--num_train_clients", type=int,
                        help="number of clients to sample during each training step",
                        default=3)
    parser.add_argument("--batches_to_take", type=int,
                        help="number of batches to sample during each training step "
                             "from selected clients",
                        default=64)
    args = parser.parse_args()
    training_config = TrainingConfig(batch_size=args.batch_size, epochs=args.epochs,
                                     shuffle_buffer=args.shuffle_buffer,
                                     num_train_clients=args.num_train_clients,
                                     batches_to_take=args.batches_to_take)
    main(args.data_fp, training_config)
