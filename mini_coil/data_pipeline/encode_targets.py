
"""
Extended encode_targets.py

Nouveautés :
- --dataset : charger MS MARCO v2 (ou autre via Hugging Face)
- --wordlist : filtrer les passages selon un vocabulaire .txt
- --model-name : choisir le backbone
    * Si supporté par fastembed → utilise TextEmbedding
    * Sinon → fallback SentenceTransformer (Gemma, SBERT, etc.)
- Log clair sur l’utilisation GPU/CPU
- Forcer les embeddings à être sauvegardés en float32
"""

import argparse
import gzip
import os
from typing import Iterable

import numpy as np
import tqdm
from npy_append_array import NpyAppendArray
from datasets import load_dataset

from mini_coil.settings import DATA_DIR

# --- FastEmbed
from fastembed import TextEmbedding

# --- HuggingFace / SentenceTransformers
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch


def load_dataset_hf(name: str, max_count=None):
    """Charger un dataset depuis Hugging Face (ici ms_marco v2.1)."""
    if name == "msmarco_v2":
        ds = load_dataset("microsoft/ms_marco", "v2.1", split="train")
        for i, ex in enumerate(ds):
            for passage in ex["passages"]["passage_text"]:
                yield f"{ex['query_id']}\t{passage}"
            if max_count and i >= max_count:
                break
    else:
        raise ValueError(f"Dataset non supporté: {name}")


def read_texts(path: str, vocab=None, max_count=None) -> Iterable[str]:
    """Lire un fichier TSV.gz déjà préparé."""
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            line = line.strip()
            try:
                _id, sentence = line.split("\t", 1)
            except ValueError:
                continue
            if vocab:
                tokens = set(sentence.lower().split())
                if not (tokens & vocab):
                    continue
            yield sentence
            if max_count and i >= max_count:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        help="Nom du dataset (ex: msmarco_v2). Si None, utiliser --input-file.")
    parser.add_argument("--input-file", type=str, default=None,
                        help="Fichier TSV.gz (id<TAB>texte).")
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--wordlist", type=str, default=None,
                        help="Fichier vocab .txt (un mot par ligne).")
    parser.add_argument("--model-name", type=str,
                        default="google/embeddinggemma-300m",
                        help="Backbone à utiliser (Gemma, SBERT, etc.)")
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--device-count", type=int, default=None)
    parser.add_argument("--max-count", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)

    args = parser.parse_args()

    # Charger vocabulaire si fourni
    vocab = None
    if args.wordlist:
        with open(args.wordlist, "r", encoding="utf-8") as f:
            vocab = set([w.strip().lower() for w in f])

    # Charger dataset
    if args.dataset:
        text_iterator = load_dataset_hf(args.dataset, max_count=args.max_count)
    elif args.input_file:
        text_iterator = read_texts(args.input_file, vocab, args.max_count)
    else:
        raise ValueError("Spécifie --dataset ou --input-file")

    # Vérifier si le modèle est supporté par FastEmbed
    supported_models = TextEmbedding.list_supported_models()
    use_fastembed = args.model_name in supported_models

    # --- FASTEMBED ---
    if use_fastembed:
        print(f"⚡ Using FastEmbed model: {args.model_name}")
        model = TextEmbedding(
            model_name=args.model_name,
            cuda=args.use_cuda,
            device_ids=[i for i in range(args.device_count)] if args.device_count else None,
            lazy_load=True if args.device_count else False
        )

        if args.use_cuda:
            print("✅ FastEmbed running on GPU")
        else:
            print("⚠️ FastEmbed running on CPU")

        def embed_batch(texts):
            vectors = model.embed(texts, batch_size=args.batch_size,
                                  parallel=args.device_count)
            return np.array(vectors, dtype=np.float32)

    # --- SENTENCE TRANSFORMER / HUGGINGFACE ---
    else:
        try:
            print(f"⚡ Using SentenceTransformer model: {args.model_name}")
            device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
            model = SentenceTransformer(args.model_name, device=device)

            if device == "cuda":
                print(f"✅ SentenceTransformer running on GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️ SentenceTransformer running on CPU")

            def embed_batch(texts):
                return model.encode(
                    texts,
                    batch_size=args.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=False
                ).astype(np.float32)

        except Exception as e:
            print(f"⚠️ SentenceTransformer failed, fallback to HuggingFace. Reason: {e}")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            model = AutoModel.from_pretrained(args.model_name)
            if args.use_cuda and torch.cuda.is_available():
                model = model.cuda()
                print(f"✅ HuggingFace model on GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️ HuggingFace model on CPU")
            model.eval()

            def embed_batch(texts):
                batch_embeddings = []
                for i in range(0, len(texts), args.batch_size):
                    batch = texts[i:i + args.batch_size]
                    tokens = tokenizer(batch, padding=True, truncation=True,
                                       return_tensors="pt", max_length=512)
                    if args.use_cuda and torch.cuda.is_available():
                        tokens = {k: v.cuda() for k, v in tokens.items()}
                    with torch.no_grad():
                        outputs = model(**tokens)
                        # mean pooling
                        emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    batch_embeddings.append(emb.astype(np.float32))
                return np.vstack(batch_embeddings)

    # Créer output
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    text_np_emb_file = NpyAppendArray(args.output_file, delete_if_exists=True)

    buffer = []
    for sentence in tqdm.tqdm(text_iterator):
        buffer.append(sentence)
        if len(buffer) >= args.batch_size:
            vectors = embed_batch(buffer)
            for v in vectors:
                text_np_emb_file.append(v.reshape(1, -1).astype(np.float32))
            buffer = []
    # Dernier batch
    if buffer:
        vectors = embed_batch(buffer)
        for v in vectors:
            text_np_emb_file.append(v.reshape(1, -1).astype(np.float32))

    text_np_emb_file.close()
    arr = np.load(args.output_file, mmap_mode='r')
    print(f"✅ Embeddings saved: {args.output_file}, shape={arr.shape}, dtype={arr.dtype}")


if __name__ == "__main__":
    main()
