import argparse
import os
from typing import Iterable
import random
import tqdm

from mini_coil.settings import DATA_DIR


def read_abstracts(path: str) -> Iterable[str]:
    with open(path, "r") as f:
        for line in f:
            yield line.strip()


def main():
    default_input_data_path = os.path.join(DATA_DIR, "test", "bat.txt")
    default_train_path = os.path.join(DATA_DIR, "test", "train.txt")
    default_valid_path = os.path.join(DATA_DIR, "test", "valid.txt")

    default_split_ratio = 0.8

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default=default_input_data_path)
    parser.add_argument("--out-train", type=str, default=default_train_path)
    parser.add_argument('--out-valid', type=str, default=default_valid_path)
    parser.add_argument('--split-ratio', type=float, default=default_split_ratio)
    args = parser.parse_args()

    abstracts = read_abstracts(args.input_file)

    train_file = open(args.out_train, "w")
    valid_file = open(args.out_valid, "w")

    for i, abstract in tqdm.tqdm(enumerate(abstracts)):
        if random.random() < args.split_ratio:
            train_file.write(abstract + "\n")
        else:
            valid_file.write(abstract + "\n")


if __name__ == "__main__":
    main()
