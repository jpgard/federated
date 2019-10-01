"""
Train a FedEd model.
"""
import argparse

from feded.datasets import fetch_larc_dataset


def main(data_fp):
    larc = fetch_larc_dataset(data_fp)
    import ipdb;
    ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fp", help="path to data csv")
    args = parser.parse_args()
    main(**vars(args))
