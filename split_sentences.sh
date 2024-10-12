#!/bin/bash


# Split sentences in OpenWebText


ARCHIVE_DIR=data/openwebtext

# Read files with the extension openwebtext-*.txt.gz and convert them to openwebtext-*-sentences.txt.gz

for file in $ARCHIVE_DIR/openwebtext-*.txt.gz
do
  # file = data/openwebtext/openwebtext-00.txt.gz

  file_name=$(basename $file)
  # file_name = openwebtext-00.txt.gz

  # Filename without extension

  file_name_no_ext=${file_name%.txt.gz}

  python -m mini_coil.data_pipeline.split_sentences --input-file $file --output-file $ARCHIVE_DIR/${file_name_no_ext}-sentences.txt.gz &
done


wait $(jobs -p)