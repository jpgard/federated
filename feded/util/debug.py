import pandas as pd

def show_batch(dataset):
    # Print the (key, value) pairs in a single batch of the dataset.
    # via https://www.tensorflow.org/tutorials/load_data/csv
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))

def print_missing_summary(df, nonzero_only=True):
    """Print a summary of fields with missing values."""
    print("[INFO] null count summary:")
    summary = df.isnull().sum(axis=0)
    if nonzero_only:
        print(summary[summary > 0])
    else:
        print(summary)