import argparse
from typing import Iterable

from qdrant_client import QdrantClient, models
import numpy as np
import tqdm
from sentence_transformers import SentenceTransformer


def read_texts(path: str) -> Iterable[str]:
    is_tsv = path.endswith(".tsv")

    with open(path, "r") as f:
        for line in tqdm.tqdm(f):
            line = line.strip()
            label = "unknown"
            if is_tsv:
                label, line = line.split("\t", 1)

            if len(line) > 0:
                yield line, label


def embed_texts(texts: Iterable[str]) -> np.ndarray:
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
    args = parser.parse_args()

    collection_name = args.collection_name

    texts_iter = read_texts(args.input_text)

    if args.input_emb is None:
        embeddings = embed_texts(map(lambda x: x[0], read_texts(args.input_text)))
    else:
        embeddings = np.load(args.input_emb, mmap_mode='r')

    def data_iter():
        for (text, label), emb in zip(texts_iter, embeddings):
            yield models.PointStruct(
                id=hash(text) % 2 ** 31,
                vector=emb.tolist(),
                payload={"text": text, "label": label}
            )

    qdrant = QdrantClient()

    qdrant.delete_collection(collection_name)

    qdrant.create_collection(
        collection_name,
        vectors_config=models.VectorParams(
            size=len(embeddings[0]),
            distance=models.Distance.COSINE
        )
    )

    qdrant.create_payload_index(
        collection_name,
        "text",
        field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.WORD,
            lowercase=True,
        )
    )

    qdrant.upload_points(collection_name, data_iter())


if __name__ == "__main__":
    main()
