"""
Creates a tf.estimator from input data
"""

import tensorflow as tf

TRAIN_EPOCHS=1000

def get_dataset(file_path, batch_size=256):
    with open(file_path) as f:
        first_line = f.readline()
    headers = first_line.split(sep=",")
    weekly_feats = [x for x in headers if x.startswith("week_")]
    dataset = tf.data.experimental.make_csv_dataset(
        # see https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/data/experimental/make_csv_dataset
        file_path,
        batch_size=batch_size,  # Artificially small to make examples easier to show.
        label_name="label_value",
        select_columns= weekly_feats + ["label_value"],
        na_value="NaN",
        shuffle=True)
    return dataset


def input_fn():
    """
    manipulate dataset, extracting the feature dict and the label
    :param dataset:
    :return:
    """
    dataset = get_dataset("/Users/jpgard/Documents/github/federated/data/data_and_labels.csv")
    # can just return the dataset directly; tf will take care of the rest
    return dataset


# Define three numeric feature columns.
feature_columns = [
    tf.feature_column.numeric_column("week_0_n_active_days"),
    tf.feature_column.numeric_column("week_0_n_forum_views"),
    tf.feature_column.numeric_column("week_0_num_posts")
    ]


# Instantiate an estimator, passing the feature columns.
estimator = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[16, 8, 4]
)

print("[INFO] training the model; to get tensorboard run tensorboard --logdir=PATH"
      "using the provided path in a terminal")
estimator.train(input_fn=input_fn, steps=TRAIN_EPOCHS)
