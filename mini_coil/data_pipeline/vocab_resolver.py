from typing import Iterable, Tuple, List

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer


class VocabResolver:
    def __init__(self, model_repository: str = None, tokenizer: Tokenizer = None):
        self.vocab = {}
        self.auto_tokenizer = AutoTokenizer.from_pretrained(model_repository) if model_repository is not None else None
        self.tokenizer = tokenizer

    def convert_ids_to_tokens(self, token_ids: np.ndarray) -> list:
        if self.tokenizer is not None:
            return [self.tokenizer.id_to_token(token_id) for token_id in token_ids]
        return self.auto_tokenizer.convert_ids_to_tokens(token_ids)

    def vocab_size(self):
        return len(self.vocab) + 1

    def add_word(self, word):
        if word not in self.vocab:
            self.vocab[word] = len(self.vocab) + 1

    def load_vocab(self, path):
        with open(path, "r") as f:
            for line in f:
                self.add_word(line.strip())

    @classmethod
    def _reconstruct_bpe(
            self, bpe_tokens: Iterable[Tuple[int, str]]
    ) -> List[Tuple[str, List[int]]]:
        result = []
        acc = ""
        acc_idx = []

        continuing_subword_prefix = "##"
        continuing_subword_prefix_len = len(continuing_subword_prefix)

        for idx, token in bpe_tokens:

            if token.startswith(continuing_subword_prefix):
                acc += token[continuing_subword_prefix_len:]
                acc_idx.append(idx)
            else:
                if acc:
                    result.append((acc, acc_idx))
                    acc_idx = []
                acc = token
                acc_idx.append(idx)

        if acc:
            result.append((acc, acc_idx))

        return result

    def token_ids_to_vocab(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Mark known tokens (including composed tokens) with vocab ids.

        Args:
            token_ids: (seq_len) - list of ids of tokens
                Example:
                    [
                        101,  3897, 19332, 12718, 23348,
                        1010,  1996,  7151,  2296, 4845,
                        2359,  2005,  4234,  1010,  4332,
                        2871,  3191,  2062, 102
                    ]

        """

        tokens = self.convert_ids_to_tokens(token_ids)
        tokens_mapping = self._reconstruct_bpe(enumerate(tokens))

        for token, mapped_token_ids in tokens_mapping:
            vocab_id = self.vocab.get(token, 0)
            for token_id in mapped_token_ids:
                token_ids[token_id] = vocab_id

        return token_ids

    def token_ids_to_vocab_batch(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Mark known tokens (including composed tokens) with vocab ids.

        Args:
            token_ids: (batch_size, seq_len) - list of ids of tokens
                Example:
                    [
                        [101,  3897, 19332, 12718, 23348],
                        [1010,  1996,  7151,  2296, 4845],
                        [2359,  2005,  4234,  1010,  4332],
                        [2871,  3191,  2062, 102, 0]
                    ]

        """

        for i in range(token_ids.shape[0]):
            self.token_ids_to_vocab(token_ids[i])

        return token_ids

    def filter(
            self,
            token_ids: np.ndarray,
            token_embeddings: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter out tokens that are not in the vocab.

        Args:
            token_ids: (batch_size, seq_len) - list of ids of tokens
            token_embeddings: (batch_size, seq_len, embedding_size) - embeddings of tokens

        Returns:
            - number of tokens in each sequence - (batch_size)
            - filtered and flattened token_ids - (total_tokens_size)
            - filtered and flattened token_embeddings - (total_tokens_size, embedding_size)
        """

        # (batch_size, seq_len)
        filtered_token_ids = self.token_ids_to_vocab_batch(token_ids)

        # (batch_size, seq_len)
        mask = filtered_token_ids.__ne__(0)

        # (batch_size)
        num_tokens = mask.sum(axis=1)

        # (total_tokens_size)
        filtered_token_ids = filtered_token_ids[mask]

        # (total_tokens_size, embedding_size)
        filtered_token_embeddings = token_embeddings[mask]

        return num_tokens, filtered_token_ids, filtered_token_embeddings


def main():
    resolver = VocabResolver()

    resolver.add_word("bat")
    resolver.add_word("nicolls")

    token_ids = np.array([
        101, 3897, 19332, 12718, 23348,
        1010, 1996, 7151, 2296, 4845,
        2359, 2005, 4234, 1010, 4332,
        2871, 3191, 2062, 102
    ])

    token_ids = resolver.token_ids_to_vocab(token_ids)

    expected = np.array([0, 0, 2, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    assert np.all(np.equal(token_ids, expected))

    batch = np.array([
        [101, 3897, 19332, 12718, 23348],
        [1010, 1996, 7151, 2296, 4845],
        [2359, 2005, 4234, 1010, 4332],
        [2871, 3191, 2062, 102, 0]
    ])

    batch = resolver.token_ids_to_vocab_batch(batch)

    expected = np.array([
        [0, 0, 2, 2, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    assert np.all(np.equal(batch, expected))


if __name__ == "__main__":
    main()
