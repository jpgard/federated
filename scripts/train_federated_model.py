"""
Trains a federated model and prints the results to console.
"""


import collections
import warnings
from six.moves import range
import numpy as np
import six
import tensorflow as tf

warnings.simplefilter('ignore')

tf.compat.v1.enable_v2_behavior()

import tensorflow_federated as tff

np.random.seed(0)

NUM_CLIENTS = 10

# NOTE: If the statement below fails, it means that you are
# using an older version of TFF without the high-performance
# executor stack. Call `tff.framework.set_default_executor()`
# instead to use the default reference runtime.
if six.PY3:
  tff.framework.set_default_executor(
      tff.framework.create_local_executor(NUM_CLIENTS))

tff.federated_computation(lambda: 'Hello, World!')()

# load example data
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

# len(emnist_train.client_ids)
# emnist_train.output_types, emnist_train.output_shapes

# create an example dataset for a single client
example_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0])
# fetch one element and show its label as numpy
example_element = iter(example_dataset).next()

example_element['label'].numpy()

# from matplotlib import pyplot as plt
# plt.imshow(example_element['pixels'].numpy(), cmap='gray', aspect='equal')
# plt.grid('off')
# _ = plt.show()


# preprocessing - this step uses the tf.Dataset and applies Dataset transformations.
# Note that features are renamed x and y for use with keras.
NUM_EPOCHS = 10
BATCH_SIZE = 20
SHUFFLE_BUFFER = 500

def preprocess(dataset):
  def element_fn(element):
    return collections.OrderedDict([
        ('x', tf.reshape(element['pixels'], [-1])),
        ('y', tf.reshape(element['label'], [1])),
    ])
  return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
      SHUFFLE_BUFFER).batch(BATCH_SIZE)

preprocessed_example_dataset = preprocess(example_dataset)

sample_batch = tf.nest.map_structure(
    lambda x: x.numpy(), iter(preprocessed_example_dataset).next())

sample_batch

# a simple helper function that will construct a list of datasets
# from the given set of users as an input to a round of training or evaluation.
# Here, each element of the list will be a tf.Dataset, but tff will also
# accept each element as a list itself, according to the docs.

def make_federated_data(client_data, client_ids):
  return [preprocess(client_data.create_tf_dataset_for_client(x))
          for x in client_ids]


# take a fixed random sample of clients, and use this for training
# NOTE: in our simulations, we can start by taking data from all
# clients at every step.
sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]

federated_train_data = make_federated_data(emnist_train, sample_clients)

len(federated_train_data), federated_train_data[0]

# create a simple keras model
def create_compiled_keras_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            10, activation=tf.nn.softmax, kernel_initializer='zeros', input_shape=(784,))])
    # compile the model
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model

def model_fn():
  keras_model = create_compiled_keras_model()
  return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

iterative_process = tff.learning.build_federated_averaging_process(model_fn)

str(iterative_process.initialize.type_signature)

# invoke the initialize computation to construct the server state.

state = iterative_process.initialize()

# run a single round of training and visualize the results,
# using the federated data we've already generated above for a sample of users.

state, metrics = iterative_process.next(state, federated_train_data)
print('round  1, metrics={}'.format(metrics))

# run a few more rounds. As noted earlier, typically at this point
# you would pick a subset of your simulation data from a new randomly
# selected sample of users for each round in order to simulate a
# realistic deployment in which users continuously come and go,
# but in this interactive notebook, for the sake of demonstration
# we'll just reuse the same users, so that the system converges quickly.

for round_num in range(2, 11):
  state, metrics = iterative_process.next(state, federated_train_data)
  print('round {:2d}, metrics={}'.format(round_num, metrics))

 # Construct a federated computation for evaluation

evaluation = tff.learning.build_federated_evaluation(model_fn)
str(evaluation.type_signature)

# invoke evaluation on the latest state we arrived at during training
train_metrics = evaluation(state.model, federated_train_data)
str(train_metrics)

# compile a test sample of federated data and rerun evaluation on the test data.
# The data will come from the same sample of real users, but from a distinct
# held-out data set.

federated_test_data = make_federated_data(emnist_test, sample_clients)

len(federated_test_data), federated_test_data[0]

test_metrics = evaluation(state.model, federated_test_data)
str(test_metrics)
