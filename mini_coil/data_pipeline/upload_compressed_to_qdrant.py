import argparse
import hashlib
import os
from typing import Iterable
import json
import itertools

from qdrant_client import QdrantClient, models
import numpy as np

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")


def load_sentences(path: str) -> Iterable[dict]:
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences-path", type=str, default=None)
    parser.add_argument("--compressed-path", type=str)
    parser.add_argument("--collection-name", type=str, default="coil-targets")
    parser.add_argument("--recreate-collection", action="store_true")
    parser.add_argument("--word", type=str)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    vectors = np.load(args.compressed_path)

    dim = vectors.shape[1]

    collection_name = args.collection_name

    collection_exists = client.collection_exists(collection_name)

    if collection_exists and args.recreate_collection:
        client.delete_collection(collection_name)
        collection_exists = False

    if not collection_exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=0,
                payload_m=16,
            )
        )

        client.create_payload_index(
            collection_name=collection_name,
            field_name="word",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
                is_tenant=True,
                on_disk=True
            )
        )

    payloads = load_sentences(args.sentences_path)

    if args.limit:
        vectors = vectors[:args.limit]
        payloads = itertools.islice(payloads, args.limit)

    client.upload_collection(
        collection_name=collection_name,
        ids=map(lambda x: hashlib.md5(args.word + x), range(len(vectors))),
        vectors=vectors,
        payload=map(lambda x: {"word": args.word, **x}, payloads),
    )


if __name__ == "__main__":
    main()
