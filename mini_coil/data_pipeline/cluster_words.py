import argparse
import json
import os
from typing import List

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_matrix(matrix_path):
    with open(matrix_path, "r") as f:
        result = json.load(f)

    offsets_row = np.array(result['offsets_row'])
    offsets_col = np.array(result['offsets_col'])
    scores = np.array(result['scores'])

    matrix = csr_matrix((scores, (offsets_row, offsets_col)))

    matrix = matrix + matrix.T

    return matrix


def plot_embeddings(embeddings, labels, save_path: str):
    import matplotlib.pyplot as plt

    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, s=1)

    plt.savefig(save_path)
    plt.close()


def find_extrema_score(scores: List[float]) -> int:
    """
    Find all scores which is an extrema of the function, meaning score before and after are lower
    """
    extrema_index = []

    for i in range(1, len(scores) - 1):
        if scores[i] > scores[i - 1] and scores[i] > scores[i + 1]:
            extrema_index.append(i)

    if len(extrema_index) == 0:
        return 0

    # return latest extrema
    return extrema_index[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-path", type=str)
    parser.add_argument("--vector-path", type=str)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    matrix_path = args.matrix_path

    matrix = load_matrix(matrix_path)

    print(matrix.shape)

    vector_path = args.vector_path
    vectors = np.load(vector_path, mmap_mode='r')

    print(vectors.shape)

    a = vectors[:, 0]
    b = vectors[:, 1]

    z = np.sqrt(1 + np.sum(vectors ** 2, axis=1))

    disk_a = a / (1 + z)
    disk_b = b / (1 + z)

    translated = np.stack([disk_a, disk_b], axis=1)

    # Inverse values of matrix to (1 - value)

    scores = []
    start_at = 2

    for i in range(start_at, 10):
        clusterer = KMeans(n_clusters=i, random_state=10)
        cluster_labels = clusterer.fit_predict(translated)

        if args.output_dir:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, f"cluster_{i}.png")
            plot_embeddings(vectors, cluster_labels, save_path)

        silhouette_avg = silhouette_score(translated, cluster_labels)  # , metric="precomputed")

        print(f"Silhouette Score for {i} clusters: {silhouette_avg}")
        scores.append(silhouette_avg)

    extrema_clusters = find_extrema_score(scores) + start_at
    print(f"Extrema extrema_clusters: {extrema_clusters}")


if __name__ == "__main__":
    main()
