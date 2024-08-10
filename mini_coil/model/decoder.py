import math

import torch

from torch import nn
from torch.nn import init


class Decoder(nn.Module):
    """
    Decoder reverses the process of the Encoder.
    It takes compressed per-word representation converts it into the context vector.
    This is similar to Autoencoder, but the intermediate layer have 2 components:
    - Compressed contextualized word vector
    - ID of the vocabulary word

    Decoder is only used during training.
    """

    def __init__(
            self,
            input_dim: int,  # Dimension of the internal representation (4)
            output_dim: int,  # Size of the context vector (768)
            vocab_size: int,  # Size of the vocabulary (10000)
            device=None,
            dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size

        # First step of the decoder
        # Per-word matrix to convert very small per-word representation to universal intermediate representation
        self.decoder_weights = nn.Parameter(torch.zeros((vocab_size, input_dim, output_dim), **factory_kwargs))

        self.output_activation = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.decoder_weights, a=math.sqrt(5))

    def forward(self,
                vocab_ids: torch.LongTensor,
                compressed: torch.Tensor) -> torch.Tensor:
        """
        Convert compressed representation into prediction of the context.

        Args:
            vocab_ids: (flatten_batch) - list pairs of vocab_ids
            compressed: (flatten_batch, input_dim) - compressed representation of the words
        """

        # Convert compressed representation into intermediate representation

        # # Select decoder weights according to vocab_ids

        # (flatten_batch, input_dim, intermediate_dim)
        decoder_weights = self.decoder_weights[vocab_ids]

        # Apply decoder weights to compressed representation
        # (flatten_batch, output_dim)
        output_raw = torch.einsum('bdi,bd->bi', decoder_weights, compressed)

        # Convert intermediate representation into prediction of the context
        # (flatten_batch, output_dim)
        prediction = self.output_activation(output_raw)

        return prediction


def test_decoder():
    decoder = Decoder(input_dim=4, output_dim=386, vocab_size=10000)
    vocab_ids = torch.randint(0, 10000, (10,))
    compressed = torch.randn(10, 4)
    prediction = decoder(vocab_ids, compressed)
    assert prediction.shape == (10, 386)


if __name__ == '__main__':
    test_decoder()
