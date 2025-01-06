import argparse
import os

import numpy as np
import tqdm


def get_tripletloss_item_chunked(distance_matrix, anchor_start, anchor_end, margin=0.2):
    triplets = []
    n = distance_matrix.shape[0]

    for anchor in range(anchor_start, min(anchor_end, n)):
        for positive in range(n):
            if positive == anchor:
                continue

            pos_dist = distance_matrix[anchor, positive]

            for negative in range(n):
                if negative == anchor or negative == positive:
                    continue

                neg_dist = distance_matrix[anchor, negative]
                if pos_dist + margin < neg_dist:
                    triplets.append((anchor, positive, negative))

        if len(triplets) >= 2_000_000:
            break

    return triplets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance-matrix-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--target-margin", type=float, default=0.2)
    args = parser.parse_args()

    distance_matrix = np.load(args.distance_matrix_path)
    n = distance_matrix.shape[0]

    os.makedirs(args.output_dir, exist_ok=True)

    chunk_size = n // args.chunks
    for i in tqdm.tqdm(range(0, n, chunk_size)):
        chunk_triplets = get_tripletloss_item_chunked(
            distance_matrix,
            anchor_start=i,
            anchor_end=i + chunk_size,
            margin=float(args.target_margin)
        )

        if chunk_triplets:
            output_file = os.path.join(args.output_dir, f"tripletloss_chunk_{i}.txt")
            with open(output_file, "w") as f:
                print(chunk_triplets, file=f)

            print(f"Chunk {i} saved with {len(chunk_triplets)} triplets")
