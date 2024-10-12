import argparse
import hashlib
import gzip

from typing import Iterable

import tqdm
from sentence_splitter import SentenceSplitter


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def read_abstracts(path: str) -> Iterable[str]:
    with gzip.open(path, "rt") as f:
        for line in f:
            yield line.strip()


def sentence_splitter(abstracts: Iterable[str]) -> Iterable[str]:
    splitter = SentenceSplitter(language='en')
    for abstract in abstracts:
        if len(abstract) == 0:
            continue
        abstract_hash = compute_hash(abstract)
        for sentence in splitter.split(abstract):
            yield abstract_hash, sentence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-file", type=str)
    args = parser.parse_args()

    abstracts = read_abstracts(args.input_file)

    sentences = sentence_splitter(abstracts)

    with gzip.open(args.output_file, "wt") as f:
        for abs_hash, sentence in tqdm.tqdm(sentences):
            f.write(f"{abs_hash}\t{sentence}\n")


if __name__ == "__main__":
    main()
