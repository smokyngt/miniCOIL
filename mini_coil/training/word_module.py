from random import random
from typing import Dict

import lightning as L
import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from mini_coil.model.cosine_loss import CosineLoss
from mini_coil.model.word_encoder import WordEncoder


class WordModule(L.LightningModule):
    def __init__(
            self,
            encoder: WordEncoder,
    ):
        super().__init__()
        self.encoder: WordEncoder = encoder
        self.loss = CosineLoss()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=2e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.5,
                    patience=5,
                    verbose=True,
                    threshold=1e-4,
                ),
                "monitor": "val_loss",
                # "frequency": "indicates how often the metric is updated",
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }


    def encode_decode_loss(
            self,
            batch: Dict[str, np.ndarray],
    ):
        """
        batch:
            {
                'word_embeddings': np array of shape (batch_size, embedding_size),
                'target_embeddings': np array of shape (batch_size, compressed_dim),
            }
        """

        # import ipdb; ipdb.set_trace()

        word_embeddings = torch.from_numpy(batch['word_embeddings']).to(self.device).float()
        target_embeddings = torch.from_numpy(batch['target_embeddings']).to(self.device).float()

        # (batch_size, compressed_dim)
        encoded = self.encoder(
            word_embeddings
        )

        # one-to-one mapping of the encoded to the target embeddings
        # (batch_size)
        mapping = torch.arange(encoded.size(0), device=encoded.device)

        loss = self.loss(mapping, encoded, target_embeddings)

        return loss

    def training_step(
            self,
            batch: Dict[str, np.ndarray],
            batch_idx: int
    ):
        loss = self.encode_decode_loss(batch)

        self.log("train_loss", loss)

        return loss

    def validation_step(
            self,
            batch: Dict[str, np.ndarray],
            batch_idx: int
    ):
        batch_size = batch['word_embeddings'].shape[0]
        loss = self.encode_decode_loss(batch)
        self.log("val_loss", loss, batch_size=batch_size)

        return loss
