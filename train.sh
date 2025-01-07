#! /usr/bin/env bash

set -e # exit on error
set -u # exit on using unset variable
set -o pipefail # exit on error in pipe

CURRENT_DIR=$(pwd -L)
COLLECTION_NAME=$1

. venv/bin/activate || . .venv/bin/activate || true

## Convert dataset into readable format

## Split data into sentences
#python -m mini_coil.data_pipeline.split_sentences \
#  --input-file "${CURRENT_DIR}/data/openwebtext-1920-sentences-vector.txt.gz" \
#  --output-file "${CURRENT_DIR}/data/openwebtext-sentences/openwebtext-1920-splitted-vector.txt.gz"

## Encode sentences with transformer model
#python -m  mini_coil.data_pipeline.encode_targets \
#  --input-file "${CURRENT_DIR}/data/openwebtext-1920-sentences-vector.txt.gz" \
#  --output-file "${CURRENT_DIR}/data/output/openwebtext-1920-splitted-vector-encodings"

## Upload encoded sentences to Qdrant
#python -m mini_coil.data_pipeline.upload_to_qdrant \
#  --input-emb
#  --input-text
#  --collection-name ${COLLECTION_NAME}

## Sample sentences with specified words and apply dimensionality reduction
python -m mini_coil.data_pipeline.compress_dimentions \
  --output-dir data/umap-8000-20-4d-mxbai-large \
  --sample-size 8000 --dim 4 --word "vector" --overwrite \
  --limit 80 --n_neighbours 80

echo "Compressed dimentions"
## Download sampled sentences
python -m mini_coil.data_pipeline.load_sentences \
   --word "vector" \
   --matrix-dir data/umap-8000-20-4d-mxbai-large \
   --output-dir data/umap-8000-20-4d-mxbai-large-sentences

echo "Loaded sentences"
## Encode sentences with smaller transformer model
python -m mini_coil.data_pipeline.encode_and_filter \
   --sentences-file data/umap-8000-20-4d-mxbai-large-sentences/sentences-vector.jsonl \
   --output-file data/umap-8000-20-4d-mxbai-large-input/word-emb-vector.npy \
   --word "vector" \
   --sample-size 8000

echo "Encoded sentences"
##Train encoder **for each word**
 python -m mini_coil.training.train_word \
  --embedding-path data/umap-8000-20-4d-mxbai-large-input/word-emb-vector.npy \
  --target-path data/umap-8000-20-4d-mxbai-large/compressed_matrix_vector.npy \
  --log-dir data/train_logs/log_vector \
  --output-path data/umap-8000-20-4d-mxbai-large-models/model-vector.ptch \
  --epochs 500
#  --gpu

echo "Trained encoder"
## Merge encoders for each word into a single model
python -m mini_coil.data_pipeline.combine_models \
  --models-dir "${CURRENT_DIR}/data/umap-8000-20-4d-mxbai-large-models" \
  --vocab-path "${CURRENT_DIR}/data/30k-vocab-filtered.txt" \
  --output-path "data/model_8000_4d" \
  --output-dim 4

