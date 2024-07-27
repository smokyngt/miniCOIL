import os
import glob
from typing import Iterator, Tuple
import pyarrow.parquet as pq

from bm43.settings import DATA_DIR


def read_data() -> Iterator[Tuple[str, str, str]]:
    parquet_files = glob.glob(os.path.join(DATA_DIR, '*.parquet'))

    for file in parquet_files:
        table = pq.read_table(file).to_pandas()

        for row in table.itertuples():
            yield (
                row.query,
                row.pos,
                row.neg
            )


def main():

    for query, pos, neg in read_data():
        print(query)
        print(pos)
        print(neg)
        break
        # import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
