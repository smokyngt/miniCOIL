import argparse
import os
from typing import Iterable, Tuple

import random
import numpy as np


def check_triplet(distance_matrix, anchor, positive, negative, margin):
    """
    Example:

        negative_distance = 0.8 # bigger = less similar
        positive_distance = 0.5 # smaller = more similar

        margin = 0.1

        negative_distance - positive_distance = 0.3 # more than margin, therefore True

        ---

        negative_distance = 0.5 # less similar
        positive_distance = 0.8 # more similar

        margin = 0.1

        negative_distance - positive_distance = -0.3 # less than margin, therefore False
    """

    pos_dist = distance_matrix[anchor, positive]
    neg_dist = distance_matrix[anchor, negative]

    return neg_dist - pos_dist > margin


def sample_triplets(distance_matrix: np.ndarray, margin: float) -> Iterable[Tuple[int, int, int]]:
    size = distance_matrix.shape[0]
    while True:
        x, y, z = random.sample(range(size), 3)

        dxy, dxz, dyz = distance_matrix[x, y], distance_matrix[x, z], distance_matrix[y, z]

        x_anchor_dist = abs(dxy - dxz)
        y_anchor_dist = abs(dxy - dyz)
        z_anchor_dist = abs(dxz - dyz)

        if x_anchor_dist > margin and x_anchor_dist > y_anchor_dist and x_anchor_dist > z_anchor_dist:
            anchor = x
            if dxy > dxz:
                positive = z
                negative = y
            else:
                positive = y
                negative = z
            yield anchor, positive, negative
            continue

        if y_anchor_dist > margin and y_anchor_dist > x_anchor_dist and y_anchor_dist > z_anchor_dist:
            anchor = y
            if dxy > dyz:
                positive = z
                negative = x
            else:
                positive = x
                negative = z
            yield anchor, positive, negative
            continue

        if z_anchor_dist > margin and z_anchor_dist > x_anchor_dist and z_anchor_dist > y_anchor_dist:
            anchor = z
            if dxz > dyz:
                positive = y
                negative = x
            else:
                positive = x
                negative = y
            yield anchor, positive, negative
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance-matrix-base-path", type=str, required=True)
    parser.add_argument("--distance-matrix-eval-path", type=str, required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--base-margin", type=float, default=0.2)
    parser.add_argument("--eval-margin", type=float, default=0.0)
    args = parser.parse_args()

    # Ground truth distance matrix
    base_distance_matrix = np.load(args.distance_matrix_base_path)

    # Evaluated distance matrix
    eval_distance_matrix = np.load(args.distance_matrix_eval_path)

    results = []

    # Sample triplets from the base distance matrix
    n = 0
    for anchor, positive, negative in sample_triplets(base_distance_matrix, args.base_margin):
        n += 1
        if n >= args.sample_size:
            break

        results.append(check_triplet(eval_distance_matrix, anchor, positive, negative, args.eval_margin))

    total_matches = sum(results)

    print(f"Total matches: {total_matches} out of {len(results)} ({total_matches / len(results):.2%})")
