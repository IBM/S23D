import os
from pathlib import Path

from .serialization import ITEM_SIZE_BYTES, deserialize_size, serialize_size


class BinaryFileRead:
    def __init__(self, filename):
        self.filename = str(filename)
        self.file = open(self.filename, "rb")

    def read(self):
        size = self.file.read(ITEM_SIZE_BYTES)
        while len(size) == ITEM_SIZE_BYTES:
            size = deserialize_size(size)
            item = self.file.read(size)
            if len(item) != size:
                raise RuntimeError(f"File {self.filename} is corrupted")
            yield item
            size = self.file.read(ITEM_SIZE_BYTES)

        if len(size):
            raise RuntimeError(f"File {self.filename} is corrupted {len(size)}")

    def close(self):
        self.file.close()


class BinaryFileWrite:
    def __init__(self, filename):
        self.filename = str(filename)
        self.file = open(self.filename, "wb")

    def write(self, item):
        size = serialize_size(len(item))
        self.file.write(size + item)

    def close(self):
        self.file.close()


def get_collection_glob_string():
    return "data*"


def get_collection_filename(chunk_index):
    return f"data{chunk_index:08d}.bin"


class BinaryFileCollectionWrite:
    def __init__(self, output_dir, chunk_size):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir()
        self.chunk_size = chunk_size
        self.item_index = 0
        self.chunk_index = 0
        self.file = None
        self.new_file()

    def new_file(self):
        if self.file is not None:
            self.file.close()
        filename = self.output_dir / get_collection_filename(self.chunk_index)
        self.file = BinaryFileWrite(filename)

    def write(self, item):
        if self.item_index == self.chunk_size:
            self.item_index = 0
            self.chunk_index += 1
            self.new_file()

        self.file.write(item)
        self.item_index += 1

    def close(self):
        self.file.close()


class BinaryFileCollectionRead:
    def __init__(self, input_dir):
        self.input_dir = Path(input_dir)

    def read(self):
        files = [str(i) for i in sorted(self.input_dir.glob("data*"))]
        for i in files:
            yield from BinaryFileRead(i).read()


def dataset_statistics(input_dir):
    input_dir = Path(input_dir)
    file_names = [str(i) for i in sorted(input_dir.glob("data*"))]

    n_samples = []
    for i in file_names:
        n_samples.append(sum(1 for _ in BinaryFileRead(i).read()))
    file_names = [os.path.basename(i) for i in file_names]
    return file_names, n_samples
