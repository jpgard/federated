"""
Train a FedEd model.

Usage (note the use of quites surrounding the wildcard path):

python train.py \
    --data_fp "./data/larc-split/train/train/train_4*.csv" \
    --epochs 20 \
    --batch_size 512 \
    --batches_to_take 128 \
    --num_train_clients 16 \
    --train_federated \
    --train_centralized

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
from feded.datasets import get_dataset_size
from feded.datasets.larc import LarcDataset, DEFAULT_LARC_TARGET_COLNAME
from feded.training.model import create_compiled_keras_model, ModelConfig
from feded.training import TrainingConfig
from feded.util.sampling import sample_client_ids, client_train_test_split
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
    """Make a sample batch from a dataset using the feature_layer."""
    # preprocess the dataset
    preprocessed_example_dataset = preprocess(
        dataset, feature_layer,
        DEFAULT_LARC_TARGET_COLNAME,
        num_epochs=training_config.epochs,
        shuffle_buffer=training_config.shuffle_buffer,
        batch_size=training_config.batch_size
    )
    # fetch a single sample batch from the preprocessed dataset
    sample_batch = tf.nest.map_structure(
        lambda x: x.numpy(), iter(preprocessed_example_dataset).next())
    return sample_batch


def execute_federated_training(dataset, logdir: str, training_config: TrainingConfig,
                               model_config: ModelConfig):
    """Execute a run of federated training."""
    create_tf_dataset_for_client_fn = lambda x: dataset.create_tf_dataset_for_client(
        x, training_config=training_config)
    feature_layer = dataset.make_feature_layer()

    train_client_ids, test_client_ids = client_train_test_split(
        dataset.client_ids, train_size=0.75, random_state=49853)
    feded_train = tff.simulation.ClientData.from_clients_and_fn(
        client_ids=train_client_ids,
        create_tf_dataset_for_client_fn=create_tf_dataset_for_client_fn
    )
    feded_test = tff.simulation.ClientData.from_clients_and_fn(
        client_ids=test_client_ids,
        create_tf_dataset_for_client_fn=create_tf_dataset_for_client_fn
    )
    example_dataset = dataset.create_tf_dataset_for_client(
        feded_train.client_ids[0], training_config=training_config
    )
    sample_batch = make_sample_batch(example_dataset, feature_layer)

    def model_fn():
        keras_model = create_compiled_keras_model(
            input_shape=(sample_batch['x'].shape[1],), model_config=model_config)
        return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn)
    print(iterative_process.initialize.type_signature)
    state = iterative_process.initialize()

    # Conduct the federated training.
    summary_writer = tf.summary.create_file_writer(logdir)
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
        with summary_writer.as_default():
            for name, metric in train_metrics._asdict().items():
                tf.summary.scalar(name, metric, step=i)


def execute_centralized_training(dataset, logdir: str, training_config: TrainingConfig,
                                 model_config: ModelConfig):
    target_feature = DEFAULT_LARC_TARGET_COLNAME
    batch_size = training_config.batch_size
    centralized_dataset = dataset.create_tf_dataset(training_config)
    feature_layer = dataset.make_feature_layer()
    sample_batch = make_sample_batch(centralized_dataset, feature_layer)
    keras_model = create_compiled_keras_model(
        input_shape=(sample_batch['x'].shape[1],), model_config=model_config)
    def _make_dataset_generator(dataset):
        """Create a generator to yield (x,y) batches for training the model."""
        batched_training_data = dataset.repeat(training_config.epochs).shuffle(
                training_config.shuffle_buffer).batch(
                training_config.batch_size)
        for element in batched_training_data:
            # extracts feature and label vectors from each element;
            # 'x' and 'y' names are required by keras.
            feature_vector = feature_layer(element)
            x = tf.reshape(feature_vector, [-1, feature_vector.shape[1]])
            y = tf.reshape(element[target_feature], [-1, 1])
            yield (x, y)
    dataset_generator = _make_dataset_generator(centralized_dataset)
    keras_model.fit_generator(
        generator=dataset_generator,
        steps_per_epoch=get_dataset_size(centralized_dataset)//training_config.batch_size,
        epochs=training_config.epochs,
        callbacks=[tf.keras.callbacks.TensorBoard(log_dir=logdir)]
    )


def main(data_fp: str, logdir: str, training_config: TrainingConfig,
         model_config: ModelConfig, train_federated: True, train_centralized: True):
    # fetch and preprocess the data, and construct federated datasets
    dataset = LarcDataset()
    dataset.read_data(data_fp)
    if train_federated:
        execute_federated_training(dataset, logdir, training_config, model_config)
    if train_centralized:
        execute_centralized_training(dataset, logdir, training_config, model_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fp", help="path to data csv")
    parser.add_argument("--epochs", type=int, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, help="batch size for training")
    parser.add_argument("--shuffle_buffer", type=int,
                        help="shuffle buffer; note that in-memory datasets should be "
                             "shuffled before modeling/batching (e.g. at read time) for "
                             "best results.",
                        default=100)
    parser.add_argument("--num_train_clients", type=int,
                        help="number of clients to sample during each training step",
                        default=3)
    parser.add_argument("--batches_to_take", type=int,
                        help="number of batches to sample during each training step "
                             "from selected clients",
                        default=64)
    parser.add_argument("--logdir", default="./tmp/logdir/")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--train_federated", action="store_true", default=False,
                        help="indicator for whether to train a federated model")
    parser.add_argument("--train_centralized", action="store_true", default=False,
                        help="indicator for whether to train a centralized model")
    args = parser.parse_args()
    training_config = TrainingConfig(batch_size=args.batch_size, epochs=args.epochs,
                                     shuffle_buffer=args.shuffle_buffer,
                                     num_train_clients=args.num_train_clients,
                                     batches_to_take=args.batches_to_take)
    model_config = ModelConfig(learning_rate=args.lr)
    main(args.data_fp, args.logdir, training_config, model_config, args.train_federated,
         args.train_centralized)
