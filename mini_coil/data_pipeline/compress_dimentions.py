import argparse
import os.path
import time

import numpy as np
from qdrant_client import QdrantClient, models
from scipy.sparse import csr_matrix

from mini_coil.settings import DATA_DIR

DEFAULT_SAMPLE_SIZE = 4000

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "default")


def query_qdrant_matrix_api(
        collection_name: str,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        limit: int = 20,
        word: str = None,
) -> models.SearchMatrixOffsetsResponse:
    time_start = time.time()

    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, port=80)

    response = qdrant.search_matrix_offsets(
        collection_name=collection_name,
        sample=sample_size,
        limit=limit,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="sentence",
                    match=models.MatchText(text=word)
                )
            ]
        ),
        timeout=60,
    )

    elapsed = time.time() - time_start

    print(f"Elapsed time: {elapsed}")

    return response


def compress_matrix(
        matrix: csr_matrix,
        dim: int = 2,
):
    from umap import UMAP

    n_components = dim

    umap = UMAP(
        metric="precomputed",
        n_components=n_components,
        output_metric="hyperboloid",
        n_neighbors=20,
    )

    start_time = time.time()
    compressed_matrix = umap.fit_transform(matrix)
    print(f"Umap fit_transform time: {time.time() - start_time}")
    return compressed_matrix


def closest_points(vectors: np.ndarray, vector: np.ndarray, n: int = 5):
    """
    Select top n closest points to the given vector using cosine similarity
    """
    from sklearn.metrics.pairwise import cosine_similarity

    similarities = cosine_similarity(vectors, vector.reshape(1, -1))

    indices = np.argsort(similarities, axis=0)[::-1]

    return indices[:n].flatten()


def estimate_precision(matrix: csr_matrix, compressed_vectors: np.ndarray, n: int = 100) -> float:
    precision = []

    for i in range(n):
        closest = closest_points(compressed_vectors, compressed_vectors[i], n=10)
        closest = closest[closest != i]

        precision.append(len(set(closest) & set(matrix[i].indices)) / len(closest))

    return np.mean(precision)


def plot_embeddings(embeddings, save_path: str):
    import matplotlib.pyplot as plt

    plt.scatter(embeddings[:, 0], embeddings[:, 1], s=1)
    plt.savefig(save_path)
    plt.close()


def get_matrix(collection_name: str, word: str, output_dir, sample_size: int = DEFAULT_SAMPLE_SIZE):
    retry = 0

    while retry < 3:
        try:
            result = query_qdrant_matrix_api(collection_name, word=word, sample_size=sample_size)
            offsets_row = np.array(result.offsets_row)
            offsets_col = np.array(result.offsets_col)
            scores = np.array(result.scores)

            matrix = csr_matrix((scores, (offsets_row, offsets_col)))

            # make sure that the matrix is symmetric
            matrix = matrix + matrix.T

            sparse_matrix_path = os.path.join(output_dir, f"sparse_matrix_{word}.json")
            with open(sparse_matrix_path, "w") as f:
                f.write(result.model_dump_json())

            return matrix
        except Exception as e:
            print(f"Error: {e}")
            retry += 1
            continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--word", type=str)
    parser.add_argument("--collection-name", type=str, default="coil")
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    args = parser.parse_args()

    collection_name = args.collection_name

    word = args.word

    matrix = get_matrix(collection_name, word, sample_size=args.sample_size, output_dir=args.output_dir)

    compressed_vectors = compress_matrix(matrix, dim=args.dim)

    if args.output_dir is None:
        output_dir = os.path.join(DATA_DIR, "test")
    else:
        output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, f"compressed_matrix_{word}.npy")
    np.save(path, compressed_vectors)

    if args.plot:
        compressed_vectors_2d = compressed_vectors[:, :2]

        plot_embeddings(compressed_vectors_2d, os.path.join(output_dir, f"compressed_matrix_{word}.png"))

        a = compressed_vectors[:, 0]
        b = compressed_vectors[:, 1]

        z = np.sqrt(1 + np.sum(compressed_vectors ** 2, axis=1))

        disk_a = a / (1 + z)
        disk_b = b / (1 + z)

        plot_embeddings(np.stack([disk_a, disk_b], axis=1),
                        os.path.join(output_dir, f"compressed_matrix_{word}_hyperboloid.png"))

        precision = estimate_precision(matrix, compressed_vectors)
        print(f"Precision: {precision}")

        # precision_2d = estimate_precision(matrix, compressed_vectors_2d)
        # print(f"Precision 2d: {precision_2d}")


if __name__ == "__main__":
    main()
