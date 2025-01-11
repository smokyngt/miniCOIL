import random
from typing import Iterable, Tuple, Dict
import numpy as np


def sample_triplets(distance_matrix: np.ndarray, margin: float) -> Iterable[Tuple[int, int, int, float]]:
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
            yield anchor, positive, negative, x_anchor_dist
            continue

        if y_anchor_dist > margin and y_anchor_dist > x_anchor_dist and y_anchor_dist > z_anchor_dist:
            anchor = y
            if dxy > dyz:
                positive = z
                negative = x
            else:
                positive = x
                negative = z
            yield anchor, positive, negative, y_anchor_dist
            continue

        if z_anchor_dist > margin and z_anchor_dist > x_anchor_dist and z_anchor_dist > y_anchor_dist:
            anchor = z
            if dxz > dyz:
                positive = y
                negative = x
            else:
                positive = x
                negative = y
            yield anchor, positive, negative, z_anchor_dist
            continue


class TripletDataloader:

    def __init__(
            self,
            embeddings: np.ndarray,
            similarity_matrix: np.ndarray,
            min_margin: float = 0.1,
            batch_size: int = 32,
            # Subset of the similarity matrix to use
            range_from: int = 0,
            range_to: int = None,
            epoch_size: int = 3200
    ):
        self.embeddings = embeddings

        self.min_margin = min_margin
        self.batch_size = batch_size
        self.range_from = range_from
        self.range_to = range_to or similarity_matrix.shape[0]
        self.distance_matrix = 1.0 - similarity_matrix[self.range_from:self.range_to, self.range_from:self.range_to]
        self.epoch_size = epoch_size

    def __iter__(self) -> Iterable[Dict[str, np.ndarray]]:
        embeddings = []
        triplets = []
        margins = []

        n = 0
        for anchor, positive, negative, margin in sample_triplets(self.distance_matrix, self.min_margin):
            length = len(embeddings)

            n += 1
            if n >= self.epoch_size:
                break

            triplets.append((length, length + 1, length + 2))

            embeddings.append(self.embeddings[anchor + self.range_from])
            embeddings.append(self.embeddings[positive + self.range_from])
            embeddings.append(self.embeddings[negative + self.range_from])

            margins.append(margin)

            if len(triplets) >= self.batch_size:
                yield {
                    "embeddings": np.array(embeddings),
                    "triplets": np.array(triplets),
                    "margins": np.array(margins)
                }
                embeddings = []
                triplets = []
                margins = []

def test_triplet_dataloader():
    embeddings = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    similarity_matrix = np.array([
        [1, 0.5, 0.1, 0.2],
        [0.5, 1, 0.3, 0.4],
        [0.1, 0.3, 1, 0.5],
        [0.2, 0.4, 0.5, 1]
    ])

    dataloader = TripletDataloader(embeddings, similarity_matrix, min_margin=0.1, batch_size=2, range_from=0,
                                   range_to=4)

    batch = next(iter(dataloader))

    print(batch["embeddings"])
    print(batch["triplets"])
    print(batch["margins"])


if __name__ == '__main__':
    test_triplet_dataloader()
