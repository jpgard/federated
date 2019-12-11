import random
from sklearn.model_selection import train_test_split


def sample_client_ids(client_ids, k, method):
    if method == "random":
        return random.sample(client_ids, k)
    else:
        raise


def client_train_test_split(client_ids, train_size: float, random_state=None):
    """Split client_ids into train_size*n and (1-train_size)*n train and test groups."""
    train_ids, test_ids = train_test_split(client_ids, train_size=train_size,
                                           random_state=random_state)
    return train_ids, test_ids
