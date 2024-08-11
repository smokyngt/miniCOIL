import os
from typing import Dict

import numpy as np

from mini_coil.settings import DATA_DIR


class PreEncodedReader:
    def __init__(self, path):
        self.path = path
        token_np_emb_file = os.path.join(self.path, "token_embeddings.npy")
        text_np_emb_file = os.path.join(self.path, "text_embeddings.npy")
        tokens_np_file = os.path.join(self.path, "tokens.npy")
        offsets_file = os.path.join(self.path, "offsets.npy")

        self.offsets = np.load(offsets_file)

        self.token_embeddings = np.load(token_np_emb_file, mmap_mode='r')
        self.text_embeddings = np.load(text_np_emb_file, mmap_mode='r')
        self.token_ids = np.load(tokens_np_file, mmap_mode='r')

        # print("self.offsets", self.offsets.shape)
        # print("self.token_embeddings", self.token_embeddings.shape)
        # print("self.text_embeddings", self.text_embeddings.shape)
        # print("self.tokens", self.tokens.shape)

    def __len__(self):
        return len(self.offsets) - 1

    def read_one(self, idx: int) -> Dict[str, np.ndarray]:
        start = self.offsets[idx]
        end = self.offsets[idx + 1]
        token_ids = self.token_ids[start:end]

        return {
            'token_embeddings': self.token_embeddings[start:end],
            'text_embeddings': self.text_embeddings[idx],
            'token_ids': token_ids
        }

    def read(self, from_idx: int, to_idx: int) -> Dict[str, np.ndarray]:
        token_ids_batch = []
        token_embeddings_batch = []
        text_embeddings_batch = []

        for idx in range(from_idx, to_idx):
            data = self.read_one(idx)

            token_ids_batch.append(data['token_ids'])
            token_embeddings_batch.append(data['token_embeddings'])
            text_embeddings_batch.append(data['text_embeddings'])

        # (batch_size, embedding_size)
        text_embeddings_batch = np.stack(text_embeddings_batch)

        # token_embeddings_batch and token_ids_batch require padding
        max_len = max(len(x) for x in token_ids_batch)
        token_embeddings_padded = np.zeros((len(token_embeddings_batch), max_len, token_embeddings_batch[0].shape[1]))
        token_ids_padded = np.zeros((len(token_ids_batch), max_len), dtype=np.int64)

        for i, (token_ids, token_embeddings) in enumerate(zip(token_ids_batch, token_embeddings_batch)):
            token_embeddings_padded[i, :len(token_ids)] = token_embeddings
            token_ids_padded[i, :len(token_ids)] = token_ids

        return {
            'token_embeddings': token_embeddings_padded,
            'text_embeddings': text_embeddings_batch,
            'token_ids': token_ids_padded
        }


def main():
    path = os.path.join(DATA_DIR, "test")
    reader = PreEncodedReader(path)
    batch = reader.read(20, 25)
    print("token_embeddings", batch['token_embeddings'].shape)
    print("text_embeddings", batch['text_embeddings'].shape)
    print("token_ids", batch['token_ids'].shape)


if __name__ == "__main__":
    main()
