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
    path = os.path.join(DATA_DIR, "test")
    input = os.path.join(path, "bat.txt")
    output = os.path.join(path, "bat_sentences.txt")

    abstracts = read_abstracts(input)

    sentences = sentence_splitter(abstracts)

    with open(output, "w") as f:
        for sentence in sentences:
            if "bat" in sentence.lower():
                f.write(sentence + "\n")


if __name__ == "__main__":
    main()
