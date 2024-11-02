import argparse
import os
from collections import defaultdict
from typing import List, Dict
import json
import tqdm

from qdrant_client import QdrantClient

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=True,
)


# Iterate over vocabulary and distance matrixes to select which abstracts and sentences should we used for inference
# Additionally save the words accosiated with the selected abstracts and sentences

def load_vocabulary(path: str) -> List[str]:
    vocabulary = []
    with open(path, 'r') as f:
        for line in f:
            vocabulary.append(line.strip())

    return vocabulary


def load_matrix_ids(directory: str, word: str):
    path = os.path.join(directory, f"sparse_matrix_{word}.json")
    if not os.path.exists(path):
        return None

    with open(path, 'r') as f:
        matrix = json.load(f)
        ids = matrix['ids']
        return ids


def get_all_id_to_abstract(collection_name: str, limit=1000) -> Dict[str, str]:
    id_to_abstract = {}
    progress_bar = tqdm.tqdm()

    points, offset = client.scroll(
        collection_name,
        with_payload=["abs_hash"],
        with_vectors=False,
        limit=limit
    )

    for point in points:
        id_to_abstract[point.id] = point.payload['abs_hash']

    progress_bar.update(len(points))

    while offset is not None:
        points_batch, offset = client.scroll(
            collection_name,
            with_payload=["abs_hash"],
            with_vectors=False,
            offset=offset,
            limit=limit
        )

        progress_bar.update(len(points_batch))

        for point in points_batch:
            id_to_abstract[point.id] = point.payload['abs_hash']

    return id_to_abstract


def load_abstract_ids(collection_name: str, ids: List[str]) -> Dict[str, str]:
    points = client.retrieve(
        collection_name,
        ids,
        with_payload=["abs_hash"],
        with_vectors=False
    )

    points_to_abstracts = dict(
        (point.id, point.payload['abs_hash'])
        for point in points
    )

    return points_to_abstracts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-path", type=str)
    parser.add_argument("--abstracts-collection-name", type=str, default="coil-abstracts")
    parser.add_argument("--sentences-collection-name", type=str, default="coil")
    parser.add_argument("--matrix-dir", type=str)
    parser.add_argument("--output-path", type=str)
    args = parser.parse_args()

    vocabulary = load_vocabulary(args.vocab_path)

    abstracts_to_words = defaultdict(list)

    sentences_collection_name = args.sentences_collection_name
    abstracts_collection_name = args.abstracts_collection_name

    ids_to_abstract = get_all_id_to_abstract(sentences_collection_name)

    for word in tqdm.tqdm(vocabulary):
        ids = load_matrix_ids(args.matrix_dir, word)
        if ids is None:
            print(f"Matrix for word {word} does not exist")
            continue
        for idx in ids:
            abstract_hash = ids_to_abstract.get(idx)
            if abstract_hash is not None:
                abstracts_to_words[abstract_hash].append(word)

    with open(args.output_path, 'w') as f:
        json.dump(abstracts_to_words, f, indent=2)


if __name__ == "__main__":
    main()
