import numpy as np
from mini_coil.data_pipeline.read_pre_encoded import PreEncodedReader
import os


class ContrastiveLoader:
    def __init__(self,
                 embeddings_path: str,
                 distance_matrix_path: str,
                 batch_size: int = 32):
        self.reader = PreEncodedReader(embeddings_path)
        self.distance_matrix = np.load(distance_matrix_path)
        self.batch_size = batch_size

    def __iter__(self):
        total_samples = len(self.reader)
        triplets_per_batch = self.batch_size // 3

        for batch_idx in range(0, total_samples - triplets_per_batch * 3, triplets_per_batch * 3):
            batch_data = self.reader.read(batch_idx, batch_idx + triplets_per_batch * 3)

            embeddings = []
            pairs = []
            labels = []

            for i in range(triplets_per_batch):
                anchor_idx = i * 3
                pos_idx = i * 3 + 1
                neg_idx = i * 3 + 2

                embeddings.extend([
                    batch_data['token_embeddings'][anchor_idx],
                    batch_data['token_embeddings'][pos_idx],
                    batch_data['token_embeddings'][neg_idx]
                ])

                base_idx = i * 3
                pairs.extend([
                    [base_idx, base_idx + 1],
                    [base_idx, base_idx + 2]
                ])

                orig_anchor_idx = batch_idx + i * 3
                orig_pos_idx = batch_idx + i * 3 + 1
                orig_neg_idx = batch_idx + i * 3 + 2

                pos_distance = self.distance_matrix[orig_anchor_idx, orig_pos_idx]
                neg_distance = self.distance_matrix[orig_anchor_idx, orig_neg_idx]

                labels.extend([pos_distance, neg_distance])

            out = {
                'embeddings': np.array(embeddings),
                'pairs': np.array(pairs),
                'labels': np.array(labels)
            }
            yield out


if __name__ == "__main__":
    N_SAMPLES = 120
    EMBEDDING_DIM = 768
    VOCAB_SIZE = 1000
    test_dir = "test_data"
    os.makedirs(test_dir, exist_ok=True)

    token_embeddings = np.random.randn(N_SAMPLES, EMBEDDING_DIM)
    text_embeddings = np.random.randn(N_SAMPLES, EMBEDDING_DIM)
    token_offsets = np.arange(N_SAMPLES + 1) * EMBEDDING_DIM
    text_offsets = np.arange(N_SAMPLES + 1) * EMBEDDING_DIM

    tokens = np.random.randint(1, VOCAB_SIZE, size=N_SAMPLES)

    np.save(os.path.join(test_dir, "token_embeddings.npy"), token_embeddings)
    np.save(os.path.join(test_dir, "text_embeddings.npy"), text_embeddings)
    np.save(os.path.join(test_dir, "token_offsets.npy"), token_offsets)
    np.save(os.path.join(test_dir, "text_offsets.npy"), text_offsets)
    np.save(os.path.join(test_dir, "tokens.npy"), tokens)

    distance_matrix = np.random.rand(N_SAMPLES, N_SAMPLES)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)
    np.save(os.path.join(test_dir, "distance_matrix.npy"), distance_matrix)

    loader = ContrastiveLoader(
        embeddings_path=test_dir,
        distance_matrix_path=os.path.join(test_dir, "distance_matrix.npy"),
        batch_size=6
    )

    for batch_idx, batch in enumerate(loader):
        print(f"\nBatch {batch_idx}")
        print(f"Embeddings shape: {batch['embeddings'].shape}")
        print(f"Pairs shape: {batch['pairs'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        if batch_idx >= 1:
            break
