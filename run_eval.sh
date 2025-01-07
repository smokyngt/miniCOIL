#! /usr/bin/env bash

set -e # exit on error
set -u # exit on using unset variable
set -o pipefail # exit on error in pipe

mkdir -p tests/em/t || true
CURRENT_DIR=$(pwd -L)
SENTENCES_FILE="${CURRENT_DIR}/data/vector-validation-new-1000.txt"
WORD_TO_TEST="vector"

function create_embeddings() {
  echo "Creating embeddings"
  python -m tests.01_embedding_maker \
    --minicoil-test-word "${WORD_TO_TEST}" \
    --input-file "${SENTENCES_FILE}" \
    --vocab-path "${CURRENT_DIR}/data/minicoil.model.vocab" \
    --word-encoder-path "${CURRENT_DIR}/data/minicoil.model.npy" \
    --output-random "${CURRENT_DIR}/tests/em/random.npy" \
    --output-mixedbread "${CURRENT_DIR}/tests/em/mixedbread.npy" \
    --output-jina "${CURRENT_DIR}/tests/em/jina_small.npy" \
    --output-minicoil "${CURRENT_DIR}/tests/em/minicoil.npy"

}

function create_matrix() {
    echo "Creating distance matrix: Minicoil"
    python -m tests.02_matrix_create \
        --input "${CURRENT_DIR}/tests/em/minicoil.npy" \
        --output "${CURRENT_DIR}/tests/em/distance_matrix_minicoil.npy"

    python -m tests.02_matrix_create \
        --input "${CURRENT_DIR}/tests/em/mixedbread.npy" \
        --output "${CURRENT_DIR}/tests/em/distance_matrix_mixedbread.npy"

    python -m tests.02_matrix_create \
        --input "${CURRENT_DIR}/tests/em/jina_small.npy" \
        --output "${CURRENT_DIR}/tests/em/distance_matrix_jina_small.npy"

    python -m tests.02_matrix_create \
        --input "${CURRENT_DIR}/tests/em/random.npy" \
        --output "${CURRENT_DIR}/tests/em/distance_matrix_random.npy"
}

function evaluate() {
    echo "Evaluating triplets: Mixed vs Jina"
    python -m tests.03_matrix_triplets \
      --distance-matrix-base-path "${CURRENT_DIR}/tests/em/distance_matrix_mixedbread.npy" \
      --distance-matrix-eval-path "${CURRENT_DIR}/tests/em/distance_matrix_jina_small.npy" \
      --sample-size 10000 \
      --base-margin 0.1 \
      --eval-margin 0.01

    echo "Evaluating triplets: Mixed vs miniCOIL"
    python -m tests.03_matrix_triplets \
      --distance-matrix-base-path "${CURRENT_DIR}/tests/em/distance_matrix_mixedbread.npy" \
      --distance-matrix-eval-path "${CURRENT_DIR}/tests/em/distance_matrix_minicoil.npy" \
      --sample-size 10000 \
      --base-margin 0.1 \
      --eval-margin 0.01

    echo "Evaluating triplets: Jina vs miniCOIL"
    python -m tests.03_matrix_triplets \
      --distance-matrix-base-path "${CURRENT_DIR}/tests/em/distance_matrix_jina_small.npy" \
      --distance-matrix-eval-path "${CURRENT_DIR}/tests/em/distance_matrix_minicoil.npy" \
      --sample-size 10000 \
      --base-margin 0.1 \
      --eval-margin 0.01


    echo "Evaluating triplets: Mixed vs Random"
    python -m tests.03_matrix_triplets \
      --distance-matrix-base-path "${CURRENT_DIR}/tests/em/distance_matrix_mixedbread.npy" \
      --distance-matrix-eval-path "${CURRENT_DIR}/tests/em/distance_matrix_random.npy" \
      --sample-size 10000 \
      --base-margin 0.1 \
      --eval-margin 0.001
}

create_embeddings
create_matrix
evaluate

# manual check (optional)
#python -m tests.matrix_check \
#  --distance-matrix "${CURRENT_DIR}/tests/em/distance_matrix_minicoil.npy" \
#  --sentences-file "${SENTENCES_FILE}" \
#  --anomalies-file ./anomalies_2025-01-06.txt \
#  --vocab-path "${CURRENT_DIR}/data/minicoil.model.vocab" \
#  --word-encoder-path "${CURRENT_DIR}/data/minicoil.model.npy" \
#  --sentence-encoder-model jinaai/jina-embeddings-v2-small-en-tokens \
#  --mixedbread-model mixedbread-ai/mxbai-embed-large-v1 \
#  --target-word "${WORD_TO_TEST}"
