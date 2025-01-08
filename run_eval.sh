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

    echo "Creating distance matrix: Mixedbread"
    python -m tests.02_matrix_create \
        --input "${CURRENT_DIR}/tests/em/mixedbread.npy" \
        --output "${CURRENT_DIR}/tests/em/distance_matrix_mixedbread.npy"

    echo "Creating distance matrix: Jina"
    python -m tests.02_matrix_create \
        --input "${CURRENT_DIR}/tests/em/jina_small.npy" \
        --output "${CURRENT_DIR}/tests/em/distance_matrix_jina_small.npy"

    echo "Creating distance matrix: Random"
    python -m tests.02_matrix_create \
        --input "${CURRENT_DIR}/tests/em/random.npy" \
        --output "${CURRENT_DIR}/tests/em/distance_matrix_random.npy"
}

function evaluate() {
    echo "Evaluating"
    python -m tests.03_matrix_triplets \
      --distance-matrix-base-path "${CURRENT_DIR}/tests/em/distance_matrix_mixedbread.npy" \
      --distance-matrix-eval-path "${CURRENT_DIR}/tests/em/distance_matrix_minicoil.npy" \
      --sample-size 10000 \
      --base-margin 0.1 \
      --eval-margin 0.01
}

function umap_compare() {
    python -m tests.04_umap_emb \
      --embeddings "${CURRENT_DIR}/tests/em/mixedbread.npy" \
      --output-umap "${CURRENT_DIR}/tests/em/mixedbread_umap.npy" \
      --umap-components 4 \
      --n-neighbors 20

    python -m tests.02_matrix_create \
            --input "${CURRENT_DIR}/tests/em/mixedbread_umap.npy" \
            --output "${CURRENT_DIR}/tests/em/distance_matrix_mixedbread_umap.npy"

    python -m tests.03_matrix_triplets \
      --distance-matrix-base-path "${CURRENT_DIR}/tests/em/distance_matrix_mixedbread.npy" \
      --distance-matrix-eval-path "${CURRENT_DIR}/tests/em/distance_matrix_mixedbread_umap.npy" \
      --sample-size 100000 \
      --base-margin 0.1 \
      --eval-margin 0.01
}

create_embeddings
create_matrix
evaluate
#umap_compare  # optional
