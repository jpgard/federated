import random


def sample_client_ids(client_ids, k, method):
    if method == "random":
        return random.sample(client_ids, k)
    else:
        raise