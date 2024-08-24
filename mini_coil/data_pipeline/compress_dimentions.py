import os.path
import time

import numpy as np
import requests
from scipy.sparse import csr_matrix

from mini_coil.settings import DATA_DIR

DEFAULT_SAMPLE_SIZE = 2000


def query_qdrant_matrix_api(
        collection_name: str,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        limit: int = 20,
        word: str = None,
):
    response = requests.post(
        f"http://localhost:6333/collections/{collection_name}/points/search/matrix/offsets",
        json={
            "sample": sample_size,
            "limit": limit,
            "filter": {
                "must": {
                    "key": "text",
                    "match": {
                        "text": word
                    }
                }
            }
        }
    )

    data = response.json()

    result = data["result"]
    elapsed = data["time"]

    print(f"Elapsed time: {elapsed}")

    print(result.keys())

    return result


def compress_matrix(
        matrix: csr_matrix
):
    from umap import UMAP

    n_components = 4

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


def main():
    collection_name = "coil"

    word = "bat"

    result = query_qdrant_matrix_api(collection_name, word=word)
    offsets_row = np.array(result["offsets_row"])
    offsets_col = np.array(result["offsets_col"])
    scores = np.array(result["scores"])
    ids = np.array(result["ids"])

    matrix = csr_matrix((scores, (offsets_row, offsets_col)))

    # make sure that the matrix is symmetric
    matrix = matrix + matrix.T

    compressed_vectors = compress_matrix(matrix)

    print(compressed_vectors.shape)

    print(compressed_vectors[:5])

    path = os.path.join(DATA_DIR, "test", f"compressed_matrix_{word}.npy")

    np.save(path, compressed_vectors)

    print([ids[x] for x in closest_points(compressed_vectors, compressed_vectors[0])], ids[0])
    print([ids[x] for x in closest_points(compressed_vectors, compressed_vectors[1])], ids[1])
    print([ids[x] for x in closest_points(compressed_vectors, compressed_vectors[2])], ids[2])
    print([ids[x] for x in closest_points(compressed_vectors, compressed_vectors[3])], ids[3])

    compressed_vectors_2d = compressed_vectors[:, :2]

    plot_embeddings(compressed_vectors_2d, os.path.join(DATA_DIR, "test", f"compressed_matrix_{word}.png"))

    a = compressed_vectors[:, 0]
    b = compressed_vectors[:, 1]

    z = np.sqrt(1 + np.sum(compressed_vectors ** 2, axis=1))

    disk_a = a / (1 + z)
    disk_b = b / (1 + z)

    plot_embeddings(np.stack([disk_a, disk_b], axis=1),
                    os.path.join(DATA_DIR, "test", f"compressed_matrix_{word}_hyperboloid.png"))

    precision = estimate_precision(matrix, compressed_vectors)
    print(f"Precision: {precision}")


if __name__ == "__main__":
    main()
