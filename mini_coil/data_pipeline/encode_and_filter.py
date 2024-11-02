import argparse
import json
from typing import List, Optional, Iterable

import numpy as np
import tqdm
from fastembed.late_interaction.token_embeddings import TokenEmbeddingsModel
from npy_append_array import NpyAppendArray

from mini_coil.data_pipeline.vocab_resolver import VocabResolver


def load_model(model_name):
    model = TokenEmbeddingsModel(model_name=model_name, cuda=True)
    return model


def read_sentences(file_path: str, limit_length: int = 4096) -> List[str]:
    with open(file_path, "r") as f:
        for line in f:
            doc = json.loads(line)
            yield doc["sentence"][:limit_length]


def encode_and_filter(model_name: Optional[str], word: str, sentences: List[str]) -> Iterable[np.ndarray]:
    model = load_model(model_name)
    vocab_resolver = VocabResolver(tokenizer=model.tokenizer)

    vocab_resolver.add_word(word)

    for embedding, sentence in zip(model.embed(sentences, batch_size=2), sentences):
        token_ids = np.array(model.tokenize([sentence])[0].ids)
        word_mask = vocab_resolver.token_ids_to_vocab(token_ids).astype(bool)

        total_tokens = np.sum(word_mask)
        if total_tokens == 0:
            yield np.zeros(embedding.shape[1])
            continue

        word_embeddings = embedding[word_mask]

        avg_embedding = np.mean(word_embeddings, axis=0)
        yield avg_embedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences-file", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--word", type=str)
    args = parser.parse_args()

    model_name = "jinaai/jina-embeddings-v2-small-en-tokens"

    input_file = args.sentences_file

    sentences = list(read_sentences(input_file, limit_length=1024))

    embeddings = encode_and_filter(
        model_name=model_name,
        word=args.word,
        sentences=sentences
    )

    output_file = args.output_file

    text_np_emb_file = NpyAppendArray(output_file, delete_if_exists=True)

    for emb in tqdm.tqdm(embeddings):
        emb_conv = emb.reshape(1, -1)
        text_np_emb_file.append(emb_conv)

    text_np_emb_file.close()

    text_np_emb_file = np.load(output_file, mmap_mode='r')

    print(f"text_np_emb_file {output_file} shape:", text_np_emb_file.shape)


if __name__ == "__main__":
    main()
