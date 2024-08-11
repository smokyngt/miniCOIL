"""
This script takes raw training data and applies initial embeddings to it.

The output of the process, for each abstract:

* List of token ids
* List of per-token embeddings
* Aggregate embedding of the abstract

This script can potentially generate huge amounts of data, so
it will write directly to disk.

"""

import os
from typing import Iterable

import numpy as np
import tqdm
from npy_append_array import NpyAppendArray

from mini_coil.pre_encoder import PreEncoder
from mini_coil.settings import DATA_DIR


def read_texts(path: str) -> Iterable[str]:
    with open(path, "r") as f:
        for line in tqdm.tqdm(f):
            line = line.strip()
            if len(line) > 0:
                yield line


def iter_batch(iterable, size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def main():
    model_repository = "sentence-transformers/all-MiniLM-L6-v2"
    model_save_path = os.path.join(DATA_DIR, "all_miniLM_L6_v2.onnx")

    test_data_path = os.path.join(DATA_DIR, "test", "bat.txt")
    batch_size = 32

    pre_encoder = PreEncoder(model_repository, model_save_path)

    total_token_emb_offset = 0
    offsets = [0]

    token_np_emb_file = NpyAppendArray(
        os.path.join(DATA_DIR, "test", "token_embeddings.npy"),
        delete_if_exists=True
    )
    text_np_emb_file = NpyAppendArray(
        os.path.join(DATA_DIR, "test", "text_embeddings.npy"),
        delete_if_exists=True
    )
    tokens_np_file = NpyAppendArray(
        os.path.join(DATA_DIR, "test", "tokens.npy"),
        delete_if_exists=True
    )
    offsets_file = os.path.join(DATA_DIR, "test", "offsets.npy")

    for batch in iter_batch(read_texts(test_data_path), batch_size):
        batch_output = pre_encoder.encode(batch)

        batch_offsets = batch_output["number_of_tokens"].tolist()

        for row_id, offset in enumerate(batch_offsets):
            global_offset = total_token_emb_offset + offset
            offsets.append(global_offset)

            token_emb = batch_output["token_embeddings"][row_id][:offset]
            token_np_emb_file.append(token_emb)

            text_emb = batch_output["text_embeddings"][row_id]
            text_emb = text_emb.reshape(1, -1)
            text_np_emb_file.append(text_emb)

            token_ids = batch_output["token_ids"][row_id][:offset]
            tokens_np_file.append(token_ids)

            total_token_emb_offset = global_offset

    offsets = np.array(offsets)

    np.save(offsets_file, offsets)

    token_np_emb_file.close()
    text_np_emb_file.close()
    tokens_np_file.close()

    print(total_token_emb_offset)


if __name__ == "__main__":
    main()
