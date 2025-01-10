import os

import numpy as np
import torch


def build_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    return 1.0 - np.matmul(normed, normed.T)


class ContrastiveLoader:
    def __init__(self,
                 embeddings: np.ndarray,
                 targets: np.ndarray,
                 batch_size: int = 32, use_cuda: bool = False):

        if use_cuda:
            embeddings = torch.from_numpy(embeddings).float().cuda()
            targets = torch.from_numpy(targets).float().cuda()
        else:
            embeddings = torch.from_numpy(embeddings).float()
            targets = torch.from_numpy(targets).float()
        self.embeddings = embeddings
        self.targets = targets
        self.batch_size = batch_size
        self.size = embeddings.shape[0]

    def __iter__(self):
        triplets_per_batch = self.batch_size // 3
        total_batches = self.size // (triplets_per_batch * 3)
        for i in range(total_batches):
            from_idx = i * triplets_per_batch * 3
            to_idx = from_idx + triplets_per_batch * 3
            emb_slice = self.embeddings[from_idx:to_idx]
            tgt_slice = self.targets[from_idx:to_idx]
            emb_out = []
            pairs = []
            labels = []
            for j in range(triplets_per_batch):
                anchor_idx = j * 3
                pos_idx = j * 3 + 1
                neg_idx = j * 3 + 2
                emb_out.append(emb_slice[anchor_idx].cpu().numpy())
                emb_out.append(emb_slice[pos_idx].cpu().numpy())
                emb_out.append(emb_slice[neg_idx].cpu().numpy())
                base_idx = j * 3
                pairs.append([base_idx, base_idx + 1])
                pairs.append([base_idx, base_idx + 2])
                pos_label = 1
                neg_label = 0
                # >= 0.5 gives best result
                if float(torch.norm(tgt_slice[anchor_idx] - tgt_slice[pos_idx])) >= 0.5:
                    pos_label = 0
                    neg_label = 1
                labels.append(pos_label)
                labels.append(neg_label)
            yield {
                "word_embeddings": np.array(emb_out),
                "pairs": np.array(pairs),
                "labels": np.array(labels)
            }


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
        batch_size=6
    )

    for batch_idx, batch in enumerate(loader):
        print(f"\nBatch {batch_idx}")
        print(f"Embeddings shape: {batch['embeddings'].shape}")
        print(f"Pairs shape: {batch['pairs'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        if batch_idx >= 1:
            break
