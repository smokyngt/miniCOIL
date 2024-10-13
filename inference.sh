#!/bin/bash

# Run inference for sentences

ARCHIVE_DIR=data/openwebtext

for file in $ARCHIVE_DIR/openwebtext-*-sentences.txt.gz
do
  # file = data/openwebtext/openwebtext-00-sentences.txt.gz

  file_name=$(basename $file)
  # file_name = openwebtext-00-sentences.txt.gz

  # Filename without extension

  file_name_no_ext=${file_name%.txt.gz}

  python -m mini_coil.data_pipeline.encode_targets \
    --input-file $file \
    --output-file $ARCHIVE_DIR/${file_name_no_ext}-emb.npy \
    --batch-size 32 \
    --device-count 4 \
    --use-cuda

done


