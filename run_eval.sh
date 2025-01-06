#! /usr/bin/env bash

set -e # exit on error
set -u # exit on using unset variable
set -o pipefail # exit on error in pipe

mkdir -p tests/em/t || true
source venv/bin/activate
CURRENT_DIR=$(pwd -L)
SENTENCES_FILE="${CURRENT_DIR}/tests/vector-valid-sample-1000.txt"
WORD_TO_TEST="vector"

function create_embeddings() {
  echo "Creating embeddings"
  python -m tests.01_embedding_maker \
    --minicoil-test-word "${WORD_TO_TEST}" \
    --input-file "${SENTENCES_FILE}" \
    --output-mixedbread "${CURRENT_DIR}/tests/em/mixedbread.npy" \
    --output-jina2 "${CURRENT_DIR}/tests/em/jina2_small.npy" \
    --output-jina2base "${CURRENT_DIR}/tests/em/jina2base.npy" \
    --output-minicoil "${CURRENT_DIR}/tests/em/minicoil.npy" \
    --vocab-path "${CURRENT_DIR}/data/minicoil.model.vocab" \
    --word-encoder-path "${CURRENT_DIR}/data/minicoil.model.npy"
}

function create_matrix() {
    echo "Creating distance matrix: Minicoil"
    python -m tests.02_matrix_create \
        --input "${CURRENT_DIR}/tests/em/minicoil.npy" \
        --output "${CURRENT_DIR}/tests/em/distance_matrix_minicoil.npy"

    python -m tests.02_matrix_create \
        --input "${CURRENT_DIR}/tests/em/mixedbread.npy" \
        --output "${CURRENT_DIR}/tests/em/distance_matrix_mixedbread.npy"

}

function generate_triplets() {
    echo "Generating triplets"
    python -m tests.03_matrix_triplets \
    --distance-matrix-path "${CURRENT_DIR}/tests/em/distance_matrix_mixedbread.npy" \
    --output-dir "${CURRENT_DIR}/tests/em/t" \
    --target-margin 0.2
}

function evaluate() {
    echo "Evaluating"
    python -m tests.04_matrix_test \
        --distance-matrix "${CURRENT_DIR}/tests/em/distance_matrix_minicoil.npy" \
        --triplets-dir "${CURRENT_DIR}/tests/em/t" \
        --save-anomalies \
        --target-margin 0.2
}

create_embeddings
create_matrix
generate_triplets
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
