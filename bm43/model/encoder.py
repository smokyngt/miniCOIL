"""
This model converts token embeddings into a compressed representation.
Each token have its own linear encoder

"""
import math
from typing import Tuple

import torch

from torch import nn
from torch.nn import init


class Encoder(nn.Module):
    """
        Encoder(768, 128, 4, 10000)

        Will look like this:

                                       Shared
                                       Linear                             Per-word
                                       Layer (768 -> 128) + Tanh          Encoder Matrix
         ┌─────────────────────┐        ┌─────┐
         │ Token Embedding(768)├───────►│     ├───┐                       (10k, 128, 4)
         └─────────────────────┘        │     │   │                          ┌─────────┐
                                        │     │   │                          │         │
         ┌─────────────────────┐        │     │   │                        ┌─┴───────┐ │
         │                     │        │     │   │                        │         │ │
         └─────────────────────┘        │     │   │  ┌────────────┐      ┌─┴───────┐ │ │      ┌─────────┐
                                        │     │   └─►│ Vector(128)├─────►│         │ │ ├─────►│Sigmoid  │
         ┌─────────────────────┐        │     │      └────────────┘      │         │ │ │      └─────────┘
         │                     │        │     │                          │         │ ├─┘
         └─────────────────────┘        │     │                          │         ├─┘
                                        │     │                          │         │
         ┌─────────────────────┐        │     │                          └─────────┘
         │                     │        │     │
         └─────────────────────┘        └─────┘

         Final liner transformation is accompanied by a non-linear activation function: Sigmoid.

         Sigmoid is used to ensure that the output is in the range [0, 1].
         It would be easier to visually interpret the output of the model, assuming that each dimension
         would need to encode a type of semantic cluster.
    """

    def __init__(
            self,
            input_dim: int,
            intermediate_dim: int,
            output_dim: int,
            vocab_size: int,
            device=None,
            dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size

        # Before training embeddings for individual words, we lower dimensionality of the original embeddings
        # using universal linear layer, shared across all words

        self.intermediate_layer = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim, **factory_kwargs),
            nn.Tanh(),
        )

        # For each word in the vocabulary, we have a linear encoder
        self.encoder_weights = nn.Parameter(torch.zeros((vocab_size, intermediate_dim, output_dim), **factory_kwargs))

        self.activation = nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.encoder_weights, a=math.sqrt(5))

    @classmethod
    def convert_vocab_ids(
            cls,
            vocab_ids: torch.LongTensor,
    ):
        """
        Convert vocab ids into unique-able format.

        Args:
            vocab_ids: (batch_size, seq_len) - list of word ids for each embedding.

        Convert each number into a pair of (value, batch_id)

            vocab_ids = [
                [7, 3, 6, 6, 2],
                [1, 2, 4, 0, 0]
            ]

        output:

            vocab_ids = [
                [
                    [7, 0],
                    [3, 0],
                    [6, 0],
                    [6, 0],
                    [2, 0],
                ],
                [
                    [1, 1],
                    [2, 1],
                    [4, 1],
                    [0, 1],
                    [0, 1],
                ]
            ]
        """
        batch_size, seq_len = vocab_ids.size()
        batch_ids = torch.arange(batch_size).unsqueeze(1).expand(batch_size, seq_len)
        return torch.stack((vocab_ids, batch_ids), dim=2)

    @classmethod
    def sum_by_vocab_ids(
            cls,
            vocab_ids: torch.LongTensor,
            embeddings: torch.Tensor,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        """
        Example:
            vocab_ids = [
                [7, 3, 6, 6, 2],
                [1, 2, 4, 0, 0]
            ]

            embeddings = [
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7, 0.8, 0.9],
                    [1.0, 1.1, 1.2],
                    [1.3, 1.4, 1.5],
                ],
                [
                    [1.6, 1.7, 1.8],
                    [1.9, 2.0, 2.1],
                    [2.2, 2.3, 2.4],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ]

        output:
            vocab_ids = [
                [7, 3, 6, 2],
                [1, 2, 4, 0]
            ]

            embeddings = [
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [1.7, 1.9, 2.1],
                    [1.3, 1.4, 1.5],
                ],
                [
                    [1.6, 1.7, 1.8],
                    [1.9, 2.0, 2.1],
                    [2.2, 2.3, 2.4],
                    [0.0, 0.0, 0.0],
                ]
            ]

        returns:
            (total_unique_words_per_batch, 2), (total_unique_words_per_batch, input_dim)

            Returns unique (vocab_id, batch_id) pairs and their corresponding sum of embeddings.
        """

        # (batch_size * seq_len, 2) - token id -> batch id
        # tensor([
        #           [7, 0],
        #           [3, 0],
        #           [6, 0],
        #           [6, 0],
        #           [2, 0],
        #           [1, 1],
        #           [2, 1],
        #           [4, 1],
        #           [0, 1],
        #           [0, 1]
        #        ])
        flattened_vocab_ids = cls.convert_vocab_ids(vocab_ids).flatten(start_dim=0, end_dim=1)

        # (batch_size * seq_len, input_dim)
        flattened_embeddings = embeddings.flatten(start_dim=0, end_dim=1)

        # Unique vocab ids per batch element
        # unique_flattened_vocab_ids - (total_unique_vocab_ids, 2)
        # inverse_indices - (batch_size * seq_len)
        unique_flattened_vocab_ids, inverse_indices = flattened_vocab_ids.unique(dim=0, return_inverse=True)

        # Sum up embeddings for each unique vocab id
        # (total_unique_vocab_ids, input_dim)
        unique_flattened_embeddings = torch.zeros(
            (unique_flattened_vocab_ids.size(0), embeddings.size(2)), device=embeddings.device)

        # Sum up embeddings for each unique vocab id
        # (total_unique_vocab_ids, input_dim)
        unique_flattened_embeddings.index_add_(0, inverse_indices, flattened_embeddings)

        return unique_flattened_vocab_ids, unique_flattened_embeddings

    def forward(
            self,
            vocab_ids: torch.LongTensor,
            embeddings: torch.Tensor,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        """
        Args:
            vocab_ids: (batch_size, seq_len) - list of word ids for each embedding.
            embeddings: (batch_size, seq_len, input_dim) - list of token embeddings, obtained from the transformer.

        Vocab ids may have duplicates. In this case embeddings should be summed up.

        Returns:
            (total_unique_words_per_batch, 2), (total_unique_words_per_batch, output_dim)

        """
        # (total_unique_vocab_ids, 2), (total_unique_vocab_ids, input_dim)
        unique_flattened_vocab_ids_and_batch_ids, unique_flattened_embeddings = \
            self.sum_by_vocab_ids(vocab_ids, embeddings)

        # Generate intermediate embeddings

        # (total_unique_vocab_ids, intermediate_dim)
        unique_flattened_embeddings = self.intermediate_layer(unique_flattened_embeddings)

        # Select which linear encoders to use for each embedding

        # Select linear encoders ids
        # (total_unique_vocab_ids)
        unique_flattened_vocab_ids = unique_flattened_vocab_ids_and_batch_ids[:, 0]

        # Select linear encoders
        # (total_unique_vocab_ids, input_dim, output_dim)
        unique_encoder_weights = self.encoder_weights[unique_flattened_vocab_ids]

        # (total_unique_vocab_ids, output_dim)
        unique_flattened_encoded = torch.einsum('bi,bio->bo', unique_flattened_embeddings, unique_encoder_weights)

        # Apply activation function
        unique_flattened_encoded = self.activation(unique_flattened_encoded)

        return unique_flattened_vocab_ids_and_batch_ids, unique_flattened_encoded
