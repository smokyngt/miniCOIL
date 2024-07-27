from typing import Dict

import os
import tqdm
import math
from snowballstemmer import stemmer as get_stemmer
import pickle

from mini_coil.settings import DATA_DIR


class IDFVocab:
    def __init__(self, idf: Dict[str, int]):
        self.idf_vocab = idf
        self.num_docs = 10_000_000

    def get_idf(self, token: str) -> float:
        num_tokens = self.idf_vocab.get(token, 0)
        return math.log(
            (self.num_docs - num_tokens + 0.5) / (num_tokens + 0.5) + 1.0
        )

    def save_vocab_pkl(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_vocab_pkl(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


class IdfConverter:

    def __init__(self):
        self.stemmer = get_stemmer("english")
        self.vocab = {}

    def add_token(self, token: str, count: int):
        stemmed_token = self.stemmer.stemWord(token)

        if stemmed_token in self.vocab:
            self.vocab[stemmed_token] += count
        else:
            self.vocab[stemmed_token] = count

    def read_vocab(self, path: str):
        with open(path) as f:
            for line in tqdm.tqdm(f):
                token, count = line.split(",")
                self.add_token(token, int(count))

    def to_idf_vocab(self) -> IDFVocab:
        return IDFVocab(self.vocab)


def main():
    converter = IdfConverter()

    vocab_path = os.path.join(DATA_DIR, "wp_word_idfs.csv")

    converter.read_vocab(vocab_path)

    idf_vocab = converter.to_idf_vocab()

    idf_vocab.save_vocab_pkl(os.path.join(DATA_DIR, "idf_vocab.pkl"))


if __name__ == "__main__":
    main()
