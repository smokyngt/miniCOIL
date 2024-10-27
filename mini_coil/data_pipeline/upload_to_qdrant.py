import argparse
import os
from typing import Iterable

from qdrant_client import QdrantClient, models
import numpy as np
import hashlib
import tqdm

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")


def read_texts(path: str) -> Iterable[str]:
    is_gz = path.endswith(".gz")

    if is_gz:
        import gzip
        with gzip.open(path, "rt") as f:
            for line in f:
                abs_hash, sentence = line.strip().split("\t")
                yield abs_hash, sentence
    else:
        with open(path, "r") as f:
            for line in f:
                abs_hash, sentence = line.strip().split("\t")
                yield abs_hash, sentence


def embed_texts(texts: Iterable[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model_repository = "mixedbread-ai/mxbai-embed-large-v1"
    model = SentenceTransformer(model_repository, trust_remote_code=True)

    texts = list(texts)

    embeddings = model.encode(texts)

    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-emb", type=str, default=None)
    parser.add_argument("--input-text", type=str)
    parser.add_argument("--collection-name", type=str, default="coil")
    parser.add_argument("--recreate-collection", action="store_true")
    parser.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()

    collection_name = args.collection_name

    if args.input_emb is None:
        embeddings = embed_texts(map(lambda x: x[0], read_texts(args.input_text)))
    else:
        embeddings = np.load(args.input_emb, mmap_mode='r')

    def data_iter():
        texts_iter = read_texts(args.input_text)
        for (abs_hash, sentence), emb in zip(texts_iter, embeddings):
            # Compute hash from the text and convert it to UUID
            hash_uuid = hashlib.md5(sentence.encode()).hexdigest()

            yield models.PointStruct(
                id=hash_uuid,
                vector=emb.tolist(),
                payload={"sentence": sentence, "abs_hash": abs_hash}
            )

    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=True,
    )

    collection_exists = qdrant.collection_exists(collection_name)

    if not collection_exists or args.recreate_collection:
        qdrant.delete_collection(collection_name)

        qdrant.create_collection(
            collection_name,
            vectors_config=models.VectorParams(
                size=len(embeddings[0]),
                distance=models.Distance.COSINE
            ),
            hnsw_config=models.HnswConfigDiff(
                m=0,
                max_indexing_threads=1,
            ),
            optimizers_config=models.OptimizersConfigDiff(
                max_segment_size=2_000_000,
            ),
            shard_number=6,
        )

        qdrant.create_payload_index(
            collection_name,
            "sentence",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase=True,
                on_disk=True,
            )
        )

    qdrant.upload_points(
        collection_name,
        points=tqdm.tqdm(data_iter()),
        parallel = args.parallel,
    )


if __name__ == "__main__":
    main()
