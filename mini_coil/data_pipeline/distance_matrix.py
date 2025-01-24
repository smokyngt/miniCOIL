"""
Sample sentences with a specific word from qdrant and build full distance matrix
"""
import json
import os
import time

from qdrant_client import QdrantClient, models
import numpy as np
import argparse

DEFAULT_SAMPLE_SIZE = 4000

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:80")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")


def query_sentences(
        collection_name: str,
        word: str,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> tuple[np.ndarray, list[dict]]:

    print(QDRANT_URL, QDRANT_API_KEY)
    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=1000,
    )


    response = qdrant.query_points(
        collection_name=collection_name,
        query=models.SampleQuery(sample=models.Sample.RANDOM),
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="sentence",
                    match=models.MatchText(text=word)
                )
            ]
        ),
        limit=sample_size,
        with_payload=True,
        with_vectors=True,
    )

    vectors = np.array([point.vector for point in response.points])
    payloads = [{
        "id": point.id,
        **point.payload
    } for point in response.points]

    return vectors, payloads

def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    Compute NxN cosine similarity matrix for N vectors.
    """
    norms = np.linalg.norm(vectors, axis=1)
    normalized = vectors / (norms[:, np.newaxis] + 1e-9)
    distances = normalized @ normalized.T
    return distances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--word", type=str)
    parser.add_argument("--collection-name", type=str, default="coil")
    parser.add_argument("--output-matrix", type=str)
    parser.add_argument("--output-sentences", type=str)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)

    args = parser.parse_args()

    start_time = time.time()

    vectors, payloads = query_sentences(args.collection_name, args.word, args.sample_size)
    elapsed_time = time.time() - start_time
    print(f"Query time: {elapsed_time}")
    distances = cosine_similarity_matrix(vectors)
    elapsed_time = time.time() - start_time
    print(f"Matrix calculation time: {elapsed_time}")

    # create directory if not exists
    os.makedirs(os.path.dirname(args.output_matrix), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_sentences), exist_ok=True)

    np.save(args.output_matrix, distances)

    with open(args.output_sentences, "w") as f:
        for payload in payloads:
            f.write(json.dumps(payload))
            f.write("\n")


if __name__ == '__main__':

    main()

    def test_distance_matrix():
        vectors = np.array([
            [1, 0, 1],
            [0, 1, 1],
            [0, 1, 0]
        ])

        distances = cosine_similarity_matrix(vectors)

        print(distances)

        assert distances.shape == (3, 3)

        assert distances[0, 0] > .9999
        assert distances[1, 1] > .9999
        assert distances[2, 2] > .9999

        assert distances[0, 2] < distances[0, 1]

    # test_distance_matrix()