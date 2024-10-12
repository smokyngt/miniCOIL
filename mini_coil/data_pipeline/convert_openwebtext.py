"""
This script converts the OpenWebText dataset to the usable format.

Default format, apparently, was created by data scientists, who have no idea how to properly store data.
So they did this:

Main file (tar)
    - folder
        - sub-archive file (xz compressed)
            - tar file
                - text file
                - text file
                - text file

This script will convert all this nonsense to the simple compressed texts file,
which is possible to decompress on the fly with the CLI tools any teapot can run.
"""

import os
import tarfile
import glob
import tqdm
import gzip
import argparse
from typing import Iterable


from mini_coil.settings import DATA_DIR


def iterate_archives(archive_dir) -> Iterable[str]:
    path_to_xz_archives = os.path.join(archive_dir, "*.xz")
    all_files = glob.glob(path_to_xz_archives)
    for archive in tqdm.tqdm(all_files):
        yield archive


def read_files_from_tar_xz(archive_path: str) -> Iterable[str]:
    with tarfile.open(archive_path, "r:xz") as tar:
        for member in tar.getmembers():
            yield tar.extractfile(member).read().decode("utf-8")


def read_texts(archive_dir) -> Iterable[str]:
    for archive in iterate_archives(archive_dir):
        for text in read_files_from_tar_xz(archive):
            yield text


def main():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--output-file", type=str, default=None)
    arg_parser.add_argument("--archive-dir", type=str, default=os.path.join(DATA_DIR, "openwebtext"))

    args = arg_parser.parse_args()

    output_file = args.output_file
    archive_dir = args.archive_dir

    # output_file = os.path.join(DATA_DIR, "openwebtext.txt.gz")

    with gzip.open(output_file, "wt") as f:
        for text in tqdm.tqdm(read_texts(archive_dir)):
            f.write(text)
            f.write("\n")


if __name__ == "__main__":
    main()
