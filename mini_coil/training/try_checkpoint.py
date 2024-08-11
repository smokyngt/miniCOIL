import os

import torch

from mini_coil.data_pipeline.pre_encoder import PreEncoder
from mini_coil.data_pipeline.vocab_resolver import VocabResolver
from mini_coil.settings import DATA_DIR
from mini_coil.training.coil_module import MiniCoil
from mini_coil.training.train import get_encoder, get_decoder


def cosine_similarity(rows_a: torch.Tensor, rows_b: torch.Tensor) -> torch.Tensor:
    """
    Compute a matrix of cosine distances between two sets of vectors.
    """
    # Normalize the vectors
    rows_a = rows_a / torch.norm(rows_a, dim=1, keepdim=True)
    rows_b = rows_b / torch.norm(rows_b, dim=1, keepdim=True)

    # Compute the cosine similarity
    return torch.mm(rows_a, rows_b.T)


def main():
    text_a = "The bat flew out of the cave."
    text_b = "He is a baseball player. He knows how to swing a bat."
    text_c = "A bat can use echolocation to navigate in the dark."

    model_repository = "sentence-transformers/all-MiniLM-L6-v2"
    model_save_path = os.path.join(DATA_DIR, "all_miniLM_L6_v2.onnx")
    test_vocab_path = os.path.join(DATA_DIR, "test", "vocab.txt")

    vocab_resolver = VocabResolver(model_repository)
    vocab_resolver.load_vocab(test_vocab_path)

    pre_encoder = PreEncoder(model_repository, model_save_path)

    result = pre_encoder.encode([text_a, text_b, text_c])

    resolved_token_ids = vocab_resolver.token_ids_to_vocab_batch(
        result["token_ids"]
    )

    text_embeddings = torch.from_numpy(result["text_embeddings"])

    print("here")

    version = "8"

    model = MiniCoil.load_from_checkpoint(
        os.path.join(DATA_DIR, "..", "lightning_logs", f"version_{version}", "checkpoints", "epoch=99-step=4300.ckpt"),
        encoder=get_encoder(vocab_resolver.vocab_size()),
        decoder=get_decoder(vocab_resolver.vocab_size()),
    )

    encoder = model.encoder
    encoder.eval()

    resolved_token_ids_torch = torch.from_numpy(resolved_token_ids).to(model.device)
    token_embeddings_torch = torch.from_numpy(result["token_embeddings"]).to(model.device).float()

    encoded = encoder(
        resolved_token_ids_torch,
        token_embeddings_torch
    )

    print(encoded[0])

    bat_tokens = encoded[1][3:]

    print(bat_tokens)

    # Matrix cosine similarity
    matrix = cosine_similarity(bat_tokens, bat_tokens)

    print("compressed similarity matrix\n", matrix)

    original_matrix = cosine_similarity(text_embeddings, text_embeddings)

    print("original similarity matrix\n", original_matrix)


if __name__ == "__main__":
    main()
