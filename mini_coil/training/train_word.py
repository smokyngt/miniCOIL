import argparse
import numpy as np

import lightning as L
import torch
from ipdb import launch_ipdb_on_exception

from mini_coil.model.word_encoder import WordEncoder
from mini_coil.training.word_module import WordModule


def get_encoder(input_dim, output_dim):
    return WordEncoder(
        input_dim=input_dim,
        output_dim=output_dim,
    )


class DataLoader:

    def __init__(self, embeddings: np.ndarray, targets: np.ndarray, batch_size: int = 200):
        self.embeddings = embeddings
        self.targets = targets
        self.batch_size = batch_size

    def __iter__(self):
        total_batches = self.embeddings.shape[0] // self.batch_size
        for i in range(total_batches):
            from_idx = i * self.batch_size
            to_idx = (i + 1) * self.batch_size
            yield {
                'word_embeddings': self.embeddings[from_idx:to_idx],
                'target_embeddings': self.targets[from_idx:to_idx],
            }


def split_train_val(embeddings: np.ndarray, target: np.ndarray, val_size=0.1):
    """
    Take last N elements of the embeddings and target as validation set.
    Records are already shuffled, so we can just take the last N elements.
    """
    val_size = int(embeddings.shape[0] * val_size)
    train_embeddings = embeddings[:-val_size]
    train_target = target[:-val_size]
    val_embeddings = embeddings[-val_size:]
    val_target = target[-val_size:]

    return train_embeddings, train_target, val_embeddings, val_target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-path", type=str)
    parser.add_argument("--target-path", type=str)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--gpu', type=int, default=-1)
    args = parser.parse_args()

    embedding = np.load(args.embedding_path)

    target = np.load(args.target_path)

    train_embeddings, train_target, val_embeddings, val_target = split_train_val(embedding, target)

    input_dim = train_embeddings.shape[1]
    output_dim = train_target.shape[1]

    encoder = get_encoder(input_dim, output_dim)

    encoder.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')

    encoder_prepared = torch.ao.quantization.prepare_qat(encoder.train())

    trainer = L.Trainer(
        max_epochs=500,
    )

    train_loader = DataLoader(train_embeddings, train_target)
    valid_loader = DataLoader(val_embeddings, val_target)

    with launch_ipdb_on_exception():
        trainer.fit(
            model=WordModule(encoder_prepared),
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader
        )


if __name__ == "__main__":
    main()
