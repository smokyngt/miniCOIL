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
    text_b = "A bat can use echolocation to navigate in the dark."
    text_c = "He is a baseball player. He knows how to swing a bat."
    text_d = "Eric Byrnes, never with an at bat in Yankee Stadium and they don't get much bigger than this one."
    text_e = "It was just a cricket bat."
    text_f = "And guess who the orphans have at bat!"

    model_repository = "sentence-transformers/all-MiniLM-L6-v2"
    model_save_path = os.path.join(DATA_DIR, "all_miniLM_L6_v2.onnx")
    test_vocab_path = os.path.join(DATA_DIR, "test", "vocab.txt")

    vocab_resolver = VocabResolver(model_repository)
    vocab_resolver.load_vocab(test_vocab_path)

    pre_encoder = PreEncoder(model_repository, model_save_path)

    texts = [text_a, text_b, text_c, text_d, text_e, text_f]

    result = pre_encoder.encode(texts)

    resolved_token_ids = vocab_resolver.token_ids_to_vocab_batch(
        result["token_ids"]
    )

    text_embeddings = torch.from_numpy(result["text_embeddings"])

    print("here")

    version = "8"

    model = MiniCoil.load_from_checkpoint(
        os.path.join(DATA_DIR, "..", "lightning_logs", f"version_{version}", "checkpoints",
                     "epoch=999-step=50000.ckpt"),
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

    # (batch size, embedding size)
    selected_token_embeddings_torch = token_embeddings_torch[resolved_token_ids_torch != 0]

    print(encoded[0])

    bat_tokens = encoded[1][len(texts):]

    print(bat_tokens)

    # Matrix cosine similarity
    matrix = cosine_similarity(bat_tokens, bat_tokens)

    print("compressed similarity matrix\n", matrix)

    original_matrix = cosine_similarity(text_embeddings, text_embeddings)

    print("original similarity matrix\n", original_matrix)

    # token_matrix = cosine_similarity(token_embeddings_torch, token_embeddings_torch)
    #
    # print("token similarity matrix\n", token_matrix)

    # Replace values in each row with it's index in sorted order

    # (batch size, batch size)
    sorted_matrix, indices = torch.sort(matrix, descending=True)
    invert_indices = torch.argsort(indices)
    print("indices\n", invert_indices)

    # (batch size, batch size)
    sorted_original_matrix, original_indices = torch.sort(original_matrix, descending=True)
    invert_original_indices = torch.argsort(original_indices)
    print("original indices\n", invert_original_indices)

if __name__ == "__main__":
    main()
