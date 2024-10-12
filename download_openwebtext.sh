#!/bin/bash


# Download OpenWebText

mkdir -p data/openwebtext

for index in {0..20}
do
  index=$(printf "%02d" $index)
  wget https://huggingface.co/datasets/Skylion007/openwebtext/resolve/main/subsets/urlsf_subset${index}.tar?download=true -O data/openwebtext/urlsf_subset${index}.tar
done



