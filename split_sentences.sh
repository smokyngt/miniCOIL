#!/bin/bash


# Split sentences in OpenWebText


ARCHIVE_DIR=data/openwebtext

# Read files with the extension openwebtext-*.txt.gz and convert them to openwebtext-*-sentences.txt.gz

for file in $ARCHIVE_DIR/openwebtext-*.txt.gz
do
  # file = data/openwebtext/openwebtext-00.txt.gz

  file_name=$(basename $file)
  # file_name = openwebtext-00.txt.gz

  # index = 00
  index=${file_name: -6:2}

  python -m mini_coil.data_pipeline.split_sentences --input-file $file --output-file $ARCHIVE_DIR/openwebtext-${index}-sentences.txt.gz &
done


wait $(jobs -p)