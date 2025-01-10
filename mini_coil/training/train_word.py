import argparse
import os

import numpy as np

import lightning as L
import torch
from ipdb import launch_ipdb_on_exception
from lightning.pytorch.loggers import CSVLogger

from mini_coil.model.word_encoder import WordEncoder
from mini_coil.training.contrastive_dataloader import ContrastiveLoader
from mini_coil.training.word_module import WordModule, ContrastiveWordModule


def get_encoder(input_dim, output_dim, dropout: float = 0.05):
    return WordEncoder(
        input_dim=input_dim,
        output_dim=output_dim,
        dropout=dropout
    )


class DataLoader:

    def __init__(
            self,
            embeddings: np.ndarray,
            targets: np.ndarray,
            batch_size: int = 200,
            use_cuda: bool = True,
    ):
        if use_cuda:
            embeddings = torch.from_numpy(embeddings).float().cuda()
            targets = torch.from_numpy(targets).float().cuda()
        else:
            embeddings = torch.from_numpy(embeddings).float()
            targets = torch.from_numpy(targets).float()
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
    parser.add_argument("--distance-matrix-path", type=str)
    parser.add_argument("--target-path", type=str)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--log-dir', type=str)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)


    args = parser.parse_args()

    embedding = np.load(args.embedding_path)

    target = np.load(args.target_path)

    train_embeddings, train_target, val_embeddings, val_target = split_train_val(embedding, target, val_size=args.val_size)

    input_dim = train_embeddings.shape[1]
    output_dim = train_target.shape[1]

    encoder_load = get_encoder(input_dim, output_dim, dropout=args.dropout)

    encoder_prepared = encoder_load

    # ToDo:
    # encoder_load.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
    # encoder_prepared = torch.ao.quantization.prepare_qat(encoder_load.train())

    if args.gpu:
        accelerator = 'auto'
    else:
        accelerator = 'cpu'
        torch.set_num_threads(1)

    trainer = L.Trainer(
        max_epochs=args.epochs,
        enable_checkpointing=False,
        logger=CSVLogger(args.log_dir),
        enable_progress_bar=True,
        accelerator=accelerator,
    )

    default_batch_size = args.batch_size
    val_batch = min(val_embeddings.shape[0] - 1, default_batch_size)

    # # OLD VERSION
    # train_loader = DataLoader(
    #     train_embeddings,
    #     train_target,
    #     batch_size=args.batch_size,
    #     use_cuda=args.gpu
    # )
    # valid_loader = DataLoader(
    #     val_embeddings,
    #     val_target,
    #     use_cuda=args.gpu,
    #     batch_size=val_batch
    # )

    train_loader = ContrastiveLoader(
        embeddings=train_embeddings,
        targets=train_target,
        batch_size=args.batch_size
    )
    valid_loader = ContrastiveLoader(
        embeddings=val_embeddings,
        targets=val_target,
        batch_size=val_batch
    )
    # with launch_ipdb_on_exception():
    trainer.fit(
        model=ContrastiveWordModule(
            encoder_prepared,
            lr=args.lr,
            factor=args.factor,
            patience=args.patience),
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader
    )

    output_dir = os.path.dirname(args.output_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    torch.save(encoder_prepared.state_dict(), args.output_path)

    # Try to read the saved model
    encoder_load = get_encoder(input_dim, output_dim)
    encoder_load.load_state_dict(torch.load(args.output_path, weights_only=True))


if __name__ == "__main__":
    main()
