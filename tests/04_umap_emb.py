import argparse

import numpy as np
from umap import UMAP


def create_umap_embeddings(output: str,
                           embeddings: np.ndarray,
                           n_components,
                           n_neighbors,
                           metric: str = 'cosine', ):

    # umap = UMAP(
    #     metric="precomputed",
    #     n_components=n_components,
    #     output_metric="hyperboloid",
    #     n_neighbors=n_neighbours,
    # )

    umap = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=42,
        output_metric="hyperboloid",
    )

    umap_embeddings = umap.fit_transform(embeddings)
    np.save(output, umap_embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, required=True,
                        help="Path to original embeddings .npy file")
    parser.add_argument("--output-umap", type=str, required=True,
                        help="Path to save UMAP embeddings")
    parser.add_argument("--umap-components", type=int,
                        help="Number of UMAP components")
    parser.add_argument("--n-neighbors", type=int,
                        help="UMAP n_neighbors parameter")

    args = parser.parse_args()

    create_umap_embeddings(
        output=args.output_umap,
        embeddings=np.load(args.embeddings),
        n_components=args.umap_components,
        n_neighbors=args.n_neighbors,
    )


if __name__ == "__main__":
    main()
