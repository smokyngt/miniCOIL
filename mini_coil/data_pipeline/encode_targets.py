"""
This script takes raw training data and applies initial embeddings to it.

The output of the process, for each abstract:

* List of token ids
* List of per-token embeddings
* Aggregate embedding of the abstract

This script can potentially generate huge amounts of data, so
it will write directly to disk.

"""

import argparse
import gzip
import itertools
import os
from typing import Iterable

import numpy as np
import tqdm
from fastembed import TextEmbedding
from npy_append_array import NpyAppendArray

from mini_coil.settings import DATA_DIR


def read_texts(path: str) -> Iterable[str]:
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            _abs_hash, sentence = line.split("\t")
            yield sentence


def main():
    input_file = "bat.txt"

    default_input_data_path = os.path.join(DATA_DIR, "test", input_file)
    default_output_file = os.path.join(DATA_DIR, "test")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default=default_input_data_path)
    parser.add_argument("--output-file", type=str, default=default_output_file)
    parser.add_argument("--model-name", type=str, default="mixedbread-ai/mxbai-embed-large-v1")
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--device-count", type=int, default=None)
    parser.add_argument("--max-count", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)

    args = parser.parse_args()

    model_name = args.model_name

    device_ids = [i for i in range(args.device_count)] if args.device_count else None

    lazy_load = True if device_ids is not None else False

    model = TextEmbedding(
        model_name=model_name,
        cuda=args.use_cuda,
        device_ids=device_ids,
        lazy_load=lazy_load
    )

    parallel = len(device_ids) if device_ids else None

    batch_size = args.batch_size

    output_file = args.output_file
    output_dir = os.path.basename(output_file)

    os.makedirs(output_dir, exist_ok=True)

    text_np_emb_file = NpyAppendArray(output_file, delete_if_exists=True)

    text_iterator = read_texts(args.input_file)

    if args.max_count:
        text_iterator = itertools.islice(text_iterator, args.max_count)

    for vector in tqdm.tqdm(model.embed(
            text_iterator,
            batch_size=batch_size,
            parallel=parallel
    )):
        # Convert to float16 and reshape from (dim,) to (1, dim)
        vector_fp16 = vector.astype(np.float16).reshape(1, -1)
        text_np_emb_file.append(vector_fp16)

    text_np_emb_file.close()

    # Check the output file shape

    text_np_emb_file = np.load(output_file, mmap_mode='r')

    print(f"text_np_emb_file {output_file} shape:", text_np_emb_file.shape)


if __name__ == "__main__":
    main()
