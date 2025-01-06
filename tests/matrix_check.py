import argparse
import datetime

import numpy as np
from fastembed import TextEmbedding

from mini_coil.model.mini_coil_inference import MiniCOIL


def cos_distance(a, b):
    return 1 - (np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

class mixedbread:
    model = None

    @classmethod
    def init_model(cls, model_name):
        cls.model = TextEmbedding(model_name=model_name)

    @classmethod
    def embed(cls, sentence):
        embeddings = cls.model.embed(sentence)
        return next(embeddings)


class minicoil:
    model = None

    @classmethod
    def init_model(cls, vocab_path, word_encoder_path, sentence_encoder_model):
        cls.model = MiniCOIL(
            vocab_path=vocab_path,
            word_encoder_path=word_encoder_path,
            sentence_encoder_model=sentence_encoder_model
        )

    @classmethod
    def embed(cls, sentence, target_word=None):
        row = cls.model.encode([sentence.strip()])[0]
        if target_word and target_word in row:
            return row[target_word]["embedding"]

        v = []
        for k in row.values():
            if k["word_id"] > 0:
                v.append(k["embedding"])
        if len(v) == 0:
            v = [[0] * cls.model.output_dim]
        return np.mean(np.array(v), axis=0)


def analyze_triplet(triplet_index, similarity_matrix, sentences, f, args):
    anchor, positive, negative = triplet_index
    matrix_pos_sim = similarity_matrix[anchor, positive]
    matrix_neg_sim = similarity_matrix[anchor, negative]
    matrix_margin = matrix_pos_sim - matrix_neg_sim
    f.write(f"\n=From similarity matrix:\n")
    f.write(f"Anchor-positive similarity: {matrix_pos_sim:.4f}\n")
    f.write(f"Anchor-negative similarity: {matrix_neg_sim:.4f}\n")
    f.write(f"Margin: {matrix_margin:.4f}\n")
    f.write(f"\n=Sentences:\n")
    f.write(f"Anchor   ({anchor}): {sentences[anchor]}\n")
    f.write(f"Positive ({positive}): {sentences[positive]}\n")
    f.write(f"Negative ({negative}): {sentences[negative]}\n")

    anchor_emb = mixedbread.embed(sentences[anchor])
    pos_emb = mixedbread.embed(sentences[positive])
    neg_emb = mixedbread.embed(sentences[negative])
    calc_pos_sim = cos_distance(anchor_emb, pos_emb)
    calc_neg_sim = cos_distance(anchor_emb, neg_emb)
    calc_margin = calc_pos_sim - calc_neg_sim
    f.write(f"\n=Recalculated with mixedbread embeddings:\n")
    f.write(f"Anchor-positive similarity: {calc_pos_sim:.4f}\n")
    f.write(f"Anchor-negative similarity: {calc_neg_sim:.4f}\n")
    f.write(f"Margin: {calc_margin:.4f}\n")

    anchor_emb_mc = minicoil.embed(sentences[anchor], args.target_word)
    pos_emb_mc = minicoil.embed(sentences[positive], args.target_word)
    neg_emb_mc = minicoil.embed(sentences[negative], args.target_word)
    calc_pos_sim_mc = cos_distance(anchor_emb_mc, pos_emb_mc)
    calc_neg_sim_mc = cos_distance(anchor_emb_mc, neg_emb_mc)
    calc_margin_mc = calc_pos_sim_mc - calc_neg_sim_mc
    f.write(f"\n=Recalculated with minicoil embeddings:\n")
    f.write(f"Anchor-positive similarity: {calc_pos_sim_mc:.4f}\n")
    f.write(f"Anchor-negative similarity: {calc_neg_sim_mc:.4f}\n")
    f.write(f"Margin: {calc_margin_mc:.4f}\n")


def parse_anomaly_line(line):
    if ',' in line and len(line.split(',')) >= 3:
        try:
            parts = [p.strip() for p in line.split(',')]
            indices = []
            for part in parts:
                if 'anchor:' in part or 'positive:' in part or 'negative:' in part:
                    value = int(part.split(':')[1])
                    indices.append(value)
            if len(indices) == 3:
                return tuple(indices)
        except ValueError:
            pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance-matrix", required=True)
    parser.add_argument("--sentences-file", required=True)
    parser.add_argument("--anomalies-file", required=True)
    parser.add_argument("--vocab-path", required=True)
    parser.add_argument("--word-encoder-path", required=True)
    parser.add_argument("--sentence-encoder-model", default="jinaai/jina-embeddings-v2-small-en-tokens")
    parser.add_argument("--mixedbread-model", default="mixedbread-ai/mxbai-embed-large-v1")
    parser.add_argument("--target-word", default=None)
    args = parser.parse_args()

    mixedbread.init_model(args.mixedbread_model)
    minicoil.init_model(args.vocab_path, args.word_encoder_path, args.sentence_encoder_model)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    output_filename = f"manual_doublecheck_{date_str}.txt"

    similarity_matrix = np.load(args.distance_matrix)
    with open(args.sentences_file) as f:
        sentences = [line for line in f]

    anomalies = []
    with open(args.anomalies_file) as af:
        for line in af:
            line = line.strip()
            if not line:
                continue
            triple = parse_anomaly_line(line)
            if triple:
                anomalies.append(triple)

    with open(output_filename, "w") as f:
        for triplet_index in anomalies:
            analyze_triplet(triplet_index, similarity_matrix, sentences, f, args)


if __name__ == "__main__":
    main()
