import dataclasses
import os
import string
from collections import defaultdict
from typing import Set, List, Dict

from mini_coil.convert_idf import IDFVocab
from mini_coil.read_data import read_data
from mini_coil.settings import DATA_DIR
from mini_coil.tokenizer import WordTokenizer
from snowballstemmer import stemmer as get_stemmer


@dataclasses.dataclass
class TripletStats:
    pos_score: float
    neg_score: float
    overlapped_pos: int
    overlapped_neg: int

    pos: List[str]
    neg: List[str]


class TripletFilter:
    @classmethod
    def load_stopwords(cls) -> Set[str]:
        path = os.path.join(DATA_DIR, 'stopwords.txt')
        with open(path) as f:
            return set(f.read().splitlines())

    def __init__(
            self,
            k: float = 1.2,
            b: float = 0.75,
            avg_len: float = 64.0,
    ):
        self.k = k
        self.b = b

        self.avg_len = avg_len

        vocab_path = os.path.join(DATA_DIR, "idf_vocab.pkl")

        self.stopwords = self.load_stopwords()
        self.tokenizer = WordTokenizer
        self.stemmer = get_stemmer("english")
        self.punctuation = set(string.punctuation)
        self.idf_vocab: IDFVocab = IDFVocab.load_vocab_pkl(vocab_path)

    def _stem(self, tokens: List[str]) -> List[str]:
        stemmed_tokens = []
        for token in tokens:
            if token in self.punctuation:
                continue

            if token in self.stopwords:
                continue

            stemmed_token = self.stemmer.stemWord(token)

            if stemmed_token:
                stemmed_tokens.append(stemmed_token)
        return stemmed_tokens

    def _term_frequency(self, tokens: List[str]) -> Dict[str, float]:
        """Calculate the term frequency part of the BM25 formula.

        (
            f(q_i, d) * (k + 1)
        ) / (
            f(q_i, d) + k * (1 - b + b * (|d| / avg_len))
        )

        Args:
            tokens (List[str]): The list of tokens in the document.

        Returns:
            Dict[int, float]: The token_id to term frequency mapping.
        """
        tf_map = {}
        counter = defaultdict(int)
        for stemmed_token in tokens:
            counter[stemmed_token] += 1

        doc_len = len(tokens)
        for stemmed_token in counter:
            num_occurrences = counter[stemmed_token]
            tf_map[stemmed_token] = num_occurrences * (self.k + 1) / (
                    num_occurrences + self.k * (1 - self.b + self.b * doc_len / self.avg_len)
            )
        return tf_map

    def tokenize_and_stem(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(text.lower())
        stemmed_tokens = self._stem(tokens)
        return stemmed_tokens

    def get_bm25_score(self, tokens: List[str]) -> float:
        tf_map = self._term_frequency(tokens)
        bm25_score = 0.0
        for token in tf_map:
            idf = self.idf_vocab.get_idf(token)
            bm25_score += idf * tf_map[token]
        return bm25_score

    def check_triplet(self, query: str, pos: str, neg: str) -> TripletStats:
        """
        Extract tokens which present in query from pos and neg

        Compare the BM25 score of pos and neg overlap with query

        If the BM25 score of pos is higher than neg, return True, else False
        """

        query_tokens = set(self.tokenize_and_stem(query))
        pos_tokens = self.tokenize_and_stem(pos)
        neg_tokens = self.tokenize_and_stem(neg)

        # Keep token count in documents
        pos_overlap = [token for token in pos_tokens if token in query_tokens]
        neg_overlap = [token for token in neg_tokens if token in query_tokens]

        pos_bm25 = self.get_bm25_score(pos_overlap)
        neg_bm25 = self.get_bm25_score(neg_overlap)

        return TripletStats(
            pos_score=pos_bm25,
            neg_score=neg_bm25,
            overlapped_pos=len(pos_overlap),
            overlapped_neg=len(neg_overlap),
            pos=pos_overlap,
            neg=neg_overlap
        )


if __name__ == "__main__":
    triplet_filter = TripletFilter()

    n = 0
    interesting = 0
    trivial = 0
    tie = 0

    for (query, pos, neg) in read_data():
        if n > 1000:
            break
        n += 1

        stats = triplet_filter.check_triplet(query, pos, neg)

        if stats.overlapped_pos == 0 or stats.overlapped_neg == 0:
            continue

        if stats.pos_score < stats.neg_score:
            interesting += 1
            print(f"Query: {query}")
            print(f"Pos: {pos}")
            print(f"Neg: {neg}")
            print(f"Pos score: {stats.pos_score}")
            print(f"Neg score: {stats.neg_score}")
            print(f"Pos: {stats.pos}")
            print(f"Neg: {stats.neg}")
            print("-------------------")

        elif stats.pos_score > stats.neg_score:
            trivial += 1
        else:
            tie += 1

    print(f"Interesting: {interesting}")
    print(f"Trivial: {trivial}")
    print(f"Tie: {tie}")