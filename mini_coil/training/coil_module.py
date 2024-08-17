from typing import Dict

import lightning as L
import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from mini_coil.model.cosine_loss import CosineLoss
from mini_coil.model.decoder import Decoder
from mini_coil.model.encoder import Encoder


class MiniCoil(L.LightningModule):
    def __init__(
            self,
            encoder: Encoder,
            decoder: Decoder,
    ):
        super().__init__()
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder
        self.loss = CosineLoss()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": ReduceLROnPlateau(
        #             optimizer,
        #             mode='min',
        #             factor=0.5,
        #             patience=50,
        #             verbose=True,
        #             threshold=1e-5,
        #         ),
        #         "monitor": "val_loss",
        #         # "frequency": "indicates how often the metric is updated",
        #         # If "monitor" references validation metrics, then "frequency" should be set to a
        #         # multiple of "trainer.check_val_every_n_epoch".
        #     },
        # }

    def encode_decode_loss(
            self,
            batch: Dict[str, np.ndarray],
    ):
        """
        batch:
            {
                'token_embeddings': np array of shape (batch_size, max_len, embedding_size),
                'text_embeddings': np array of shape (batch_size, embedding_size),
                'token_ids': np array of shape (batch_size, max_len)
            }
        """

        # import ipdb; ipdb.set_trace()

        token_ids = torch.from_numpy(batch['token_ids']).to(self.device)
        token_embeddings = torch.from_numpy(batch['token_embeddings']).to(self.device).float()
        text_embeddings = torch.from_numpy(batch['text_embeddings']).to(self.device).float()

        encoded = self.encoder(
            token_ids,
            token_embeddings
        )

        # (total_unique_words_per_batch, 2)
        # [
        #     [vocab_id1, id_in_batch1],
        #     [vocab_id2, id_in_batch1],
        #     [vocab_id3, id_in_batch2],
        # ]
        unique_flattened_vocab_ids_and_batch_ids = encoded[0]

        # (total_unique_words_per_batch, compressed_dim)
        unique_flattened_encoded = encoded[1]

        # (total_unique_words_per_batch)
        vocab_ids = unique_flattened_vocab_ids_and_batch_ids[:, 0]
        # (total_unique_words_per_batch)
        word_to_text_id = unique_flattened_vocab_ids_and_batch_ids[:, 1]

        # (total_unique_words_per_batch, embedding_size)
        decompressed = self.decoder(
            vocab_ids,
            unique_flattened_encoded,
        )

        return self.loss(word_to_text_id, decompressed, text_embeddings)

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
        batch_size = batch['token_ids'].shape[0]

        loss = self.encode_decode_loss(batch)
        self.log("val_loss", loss, batch_size=batch_size)

        return loss
