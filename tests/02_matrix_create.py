import argparse

import numpy as np


def calculate_distance_matrix(input_path, output_path):
    embeddings = np.load(input_path)
    norms = np.linalg.norm(embeddings, axis=1)
    normalized = embeddings / (norms[:, np.newaxis] + 1e-9)
    distances = 1 - normalized @ normalized.T
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)
    np.save(output_path, distances)
    print(f"Matrix shape: {distances.shape}")
    print(f"Distance range: {distances.min():.6f} to {distances.max():.6f}")
    print(f"Mean distance: {mean_dist:.6f}")
    print(f"Median distance: {median_dist:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to input embeddings .npy file')
    parser.add_argument('--output', required=True, help='Path to output similarity matrix .npy file')
    args = parser.parse_args()

    calculate_distance_matrix(args.input, args.output)
