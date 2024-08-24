import argparse
import os
from typing import Iterable
from sentence_splitter import SentenceSplitter, split_text_into_sentences

from mini_coil.settings import DATA_DIR


def read_abstracts(path: str) -> Iterable[str]:
    with open(path, "r") as f:
        for line in f:
            yield line.strip()


def sentence_splitter(abstract: Iterable[str]) -> Iterable[str]:
    splitter = SentenceSplitter(language='en')
    for abs in abstract:
        for sentence in splitter.split(abs):
            yield sentence


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-file", type=str)
    args = parser.parse_args()

    abstracts = read_abstracts(args.input_file)

    sentences = sentence_splitter(abstracts)

    with open(args.output_file, "w") as f:
        for sentence in sentences:
            f.write(sentence + "\n")


if __name__ == "__main__":
    main()
