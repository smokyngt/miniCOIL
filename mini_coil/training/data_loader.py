from mini_coil.data_pipeline.read_pre_encoded import PreEncodedReader


class PreEncodedLoader:

    def __init__(self, path: str, batch_size: int = 32):
        self.reader = PreEncodedReader(path)
        self.batch_size = batch_size

    def __iter__(self):
        total_batches = len(self.reader) // self.batch_size
        for i in range(total_batches):
            yield self.reader.read(i * self.batch_size, (i + 1) * self.batch_size)
