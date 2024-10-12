#!/bin/bash


# Unpack OpenWebText tar files

ARCHIVE_DIR=data/openwebtext

rm -rf $ARCHIVE_DIR/openwebtext_subset*

rm -f $ARCHIVE_DIR/openwebtext

for archive in $ARCHIVE_DIR/*.tar
do
  # archive = data/openwebtext/urlsf_subset00.tar

  file_name=$(basename $archive)
  # file_name = urlsf_subset00.tar

  # index = 00
  index=${file_name: -6:2}

  tar -xvf $archive -C $ARCHIVE_DIR

  mv $ARCHIVE_DIR/openwebtext $ARCHIVE_DIR/openwebtext_subset${index}

  python -m mini_coil.data_pipeline.convert_openwebtext \
    --output-file $ARCHIVE_DIR/openwebtext-${index}.txt.gz \
    --archive-dir $ARCHIVE_DIR/openwebtext_subset${index} &

done



wait $(jobs -p)