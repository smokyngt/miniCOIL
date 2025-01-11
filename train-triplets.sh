#! /usr/bin/env bash

set -e # exit on error
set -u # exit on using unset variable
set -o pipefail # exit on error in pipe

CURRENT_DIR=$(pwd -L)
#COLLECTION_NAME=$1
TARGET_WORD=vector
DIM=4
SAMPLES=8000
LMODEL=mxbai-large
IMODEL=jina-small


INPUT_DIR=data/triplets-${SAMPLES}-${LMODEL}

#echo "Generate distance matrix for given word"
python -m mini_coil.data_pipeline.distance_matrix \
       --word ${TARGET_WORD} \
       --output-matrix ${INPUT_DIR}/distance_matrix/dm-${TARGET_WORD}.npy \
       --output-sentences ${INPUT_DIR}/target_sentences/sentences-${TARGET_WORD}.jsonl \
       --sample-size ${SAMPLES}

echo "Loaded sentences"

# Encode sentences with smaller transformer model
python -m mini_coil.data_pipeline.encode_and_filter \
   --sentences-file ${INPUT_DIR}/target_sentences/sentences-${TARGET_WORD}.jsonl \
   --output-file ${INPUT_DIR}-${IMODEL}/word-emb-${TARGET_WORD}.npy \
   --word "${TARGET_WORD}" \
   --sample-size "${SAMPLES}"

echo "Encoded sentences"


#Train encoder **for each word**

python -m mini_coil.training.train_word_triplet \
  --embedding-path ${INPUT_DIR}-${IMODEL}/word-emb-${TARGET_WORD}.npy \
  --distance-matrix-path ${INPUT_DIR}/distance_matrix/dm-${TARGET_WORD}.npy \
  --output-dim ${DIM} \
  --output-path ${INPUT_DIR}-${IMODEL}-${DIM}/word-models/model-${TARGET_WORD}.ptch \
  --log-dir ${INPUT_DIR}-${IMODEL}/train_logs/log_"${TARGET_WORD}" \
  --epochs 100

echo "Trained model"

## Merge encoders for each word into a single model
python -m mini_coil.data_pipeline.combine_models \
  --models-dir ${INPUT_DIR}-${IMODEL}-${DIM}/word-models \
  --vocab-path "${CURRENT_DIR}/data/30k-vocab-filtered.txt" \
  --output-path "data/model_triplet_${SAMPLES}_${DIM}d" \
  --output-dim "${DIM}"

