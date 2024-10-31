import argparse
import gzip
import hashlib
import os
from typing import Iterable

from qdrant_client import QdrantClient, models
import tqdm

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")


def read_abstracts(path: str) -> Iterable[str]:
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            yield line.strip()


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--collection-name", type=str, default="coil")
    parser.add_argument("--recreate-collection", action="store_true")
    parser.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()

    collection_name = args.collection_name

    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=True,
    )

    def data_iter():
        abstracts = read_abstracts(args.input_file)
        for abstract in abstracts:
            abs_hash = compute_hash(abstract)

            # Compute hash from the text and convert it to UUID
            hash_uuid = hashlib.md5(abs_hash.encode()).hexdigest()

            yield models.PointStruct(
                id=hash_uuid,
                vector={},
                payload={"abstract": abstract, "abs_hash": abs_hash}
            )

    collection_exists = qdrant.collection_exists(collection_name)

    if not collection_exists or args.recreate_collection:
        qdrant.delete_collection(collection_name)

        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config={},
            optimizers_config=models.OptimizersConfigDiff(
                max_segment_size=2_000_000,
                max_optimization_threads=1,  # Run one optimization per shard
            ),
            shard_number=6,
        )

        qdrant.create_payload_index(
            collection_name,
            "abs_hash",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
                on_disk=True,
            )
        )

    qdrant.upload_points(
        collection_name,
        points=tqdm.tqdm(data_iter()),
        parallel=args.parallel,
    )


if __name__ == "__main__":
    main()
