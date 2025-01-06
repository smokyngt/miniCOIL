import argparse

import numpy as np
import torch
from fastembed.late_interaction.token_embeddings import TokenEmbeddingsModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from mini_coil.model.mini_coil_inference import MiniCOIL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input file containing sentences")
    parser.add_argument("--output-mixedbread", type=str, required=True, help="Path to the output file for mixedbread")
    parser.add_argument("--output-jina2", type=str, required=True, help="Path to the output file for jina2")
    parser.add_argument("--output-jina2base", type=str, required=True, help="Path to the output file for jina2base")
    parser.add_argument("--output-minicoil", type=str, required=True, help="Path to the output file for minicoil")
    parser.add_argument("--vocab-path", type=str, required=True, help="Path to the vocabulary file (minicoil)")
    parser.add_argument("--word-encoder-path", type=str, required=True, help="Path to the word encoder file (minicoil)")
    parser.add_argument("--use-cuda", action="store_true", default=False, help="Use CUDA for jina2base")
    parser.add_argument("--skip-mixed", action="store_true", help="Skip mixedbread")
    parser.add_argument("--skip-jina2", action="store_true", help="Skip jina2")
    parser.add_argument("--skip-jina2base", action="store_true", help="Skip jina2base")
    parser.add_argument("--skip-minicoil", action="store_true", help="Skip minicoil")
    parser.add_argument("--minicoil-test-word", type=str, required=True, help="Word to test for minicoil")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        lines = [x.strip() for x in f if len(x.strip()) > 0]

    if not args.skip_mixed:
        model_mixed = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", trust_remote_code=True)
        if args.use_cuda:
            model_mixed = model_mixed.to("cuda")
        emb_mixed = model_mixed.encode(lines, batch_size=32, show_progress_bar=False)
        np.save(args.output_mixedbread, emb_mixed)

    if not args.skip_jina2:
        model_jina2 = TokenEmbeddingsModel(model_name="jinaai/jina-embeddings-v2-small-en-tokens", threads=1)
        emb_jina2_list = []
        for line in lines:
            emb = next(model_jina2.embed([line], batch_size=1))
            emb_jina2_list.append(np.array(emb))
        shapes = [x.shape for x in emb_jina2_list]
        if len(set(shapes)) == 1:
            emb_jina2 = np.stack(emb_jina2_list)
            np.save(args.output_jina2, emb_jina2)

    if not args.skip_jina2base:
        tokenizer_jina2base = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-en")
        model_jina2base = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-base-en", attn_implementation="eager",
                                                    ignore_mismatched_sizes=True)
        if args.use_cuda:
            model_jina2base = model_jina2base.to("cuda")
        emb_jina2base_list = []
        for line in lines:
            inputs = tokenizer_jina2base([line], return_tensors="pt", padding=True, truncation=True, max_length=512)
            if args.use_cuda:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model_jina2base(**inputs)
            last_hidden_state = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            emb_jina2base_list.append(mean_embeddings.cpu().numpy()[0])
        emb_jina2base = np.stack(emb_jina2base_list)
        np.save(args.output_jina2base, emb_jina2base)

    if not args.skip_minicoil:
        model_minicoil = MiniCOIL(
            vocab_path=args.vocab_path,
            word_encoder_path=args.word_encoder_path,
            sentence_encoder_model="jinaai/jina-embeddings-v2-small-en-tokens"
        )
        emb_mc_list = []
        for line in lines:
            row = model_minicoil.encode([line])[0]
            v = []
            if args.minicoil_test_word not in row:
                for k in row.values():
                    if k["word_id"] > 0:
                        v.append(k["embedding"])
                if len(v) == 0:
                    v = [[0] * model_minicoil.output_dim]
                avg = np.mean(np.array(v), axis=0)
                emb_mc_list.append(avg)
            else:
                emb_mc_list.append(row[args.minicoil_test_word]["embedding"])
        emb_mc = np.stack(emb_mc_list)
        np.save(args.output_minicoil, emb_mc)


if __name__ == "__main__":
    main()
