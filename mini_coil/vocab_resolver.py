from typing import Iterable, Tuple, List

import torch
from transformers import AutoTokenizer


class VocabResolver:
    def __init__(self, model_repository="sentence-transformers/all-MiniLM-L6-v2"):
        self.vocab = {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_repository)

    def add_word(self, word):
        if word not in self.vocab:
            self.vocab[word] = len(self.vocab) + 1

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

    def token_ids_to_vocab(self, token_ids: torch.Tensor) -> torch.Tensor:
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

        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        tokens_mapping = self._reconstruct_bpe(enumerate(tokens))

        for token, mapped_token_ids in tokens_mapping:
            vocab_id = self.vocab.get(token, 0)
            for token_id in mapped_token_ids:
                token_ids[token_id] = vocab_id

        return token_ids

    def token_ids_to_vocab_batch(self, token_ids: torch.Tensor) -> torch.Tensor:
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

        for i in range(token_ids.size(0)):
            self.token_ids_to_vocab(token_ids[i])

        return token_ids


def main():
    resolver = VocabResolver()

    resolver.add_word("bat")
    resolver.add_word("nicolls")

    token_ids = torch.tensor([
        101, 3897, 19332, 12718, 23348,
        1010, 1996, 7151, 2296, 4845,
        2359, 2005, 4234, 1010, 4332,
        2871, 3191, 2062, 102
    ])

    token_ids = resolver.token_ids_to_vocab(token_ids)

    expected = torch.tensor([0, 0, 2, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    assert torch.equal(token_ids, expected)

    batch = torch.tensor([
        [101, 3897, 19332, 12718, 23348],
        [1010, 1996, 7151, 2296, 4845],
        [2359, 2005, 4234, 1010, 4332],
        [2871, 3191, 2062, 102, 0]
    ])

    batch = resolver.token_ids_to_vocab_batch(batch)

    expected = torch.tensor([
        [0, 0, 2, 2, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    assert torch.equal(batch, expected)


if __name__ == "__main__":
    main()
