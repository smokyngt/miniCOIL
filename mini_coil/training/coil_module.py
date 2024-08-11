from typing import Dict

import lightning as L
import numpy as np
import torch
from torch import optim

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

    def training_step(
            self,
            batch: Dict[str, np.ndarray],
            batch_idx: int
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

        loss = self.loss(word_to_text_id, decompressed, text_embeddings)

        self.log("loss", loss)

        return loss
