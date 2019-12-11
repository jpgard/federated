def show_batch(dataset):
    # Print the (key, value) pairs in a single batch of the dataset.
    # via https://www.tensorflow.org/tutorials/load_data/csv
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))