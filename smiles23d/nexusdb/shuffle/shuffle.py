import heapq
from functools import partial, total_ordering
from pathlib import Path

import numpy as np

from .io import (
    BinaryFileCollectionRead,
    BinaryFileCollectionWrite,
    BinaryFileRead,
    BinaryFileWrite,
    dataset_statistics,
    get_collection_filename,
)


class RandomInt:
    def __init__(self, n, buffer_size=10000):
        self.buffer_size = buffer_size
        self.reset(n)

    def reset(self, n):
        self.n = n
        self._reset()

    def _reset(self):
        self.buffer = np.random.randint(self.n, size=self.buffer_size)
        self.pointer = 0

    def get_val(self):
        if self.pointer == self.buffer_size:
            self._reset()
        item = self.buffer[self.pointer]
        self.pointer += 1
        return item


def shuffle_file(filename):
    file = BinaryFileRead(filename)
    items = list(file.read())
    index = np.arange(len(items))
    np.random.shuffle(index)
    for i in index:
        yield items[i]
    file.close()


def sort_file(filename, schema, key):
    file = BinaryFileRead(filename)
    items = list(file.read())
    file.close()
    keys = [key(schema.deserialize(item)) for item in items]
    keys = np.argsort(keys)
    for i in keys:
        yield items[i]


def _shuffle_each_collection_file(input_dir, output_dir, shuffle_func):
    input_dir = Path(input_dir)
    files = [str(i) for i in sorted(input_dir.glob("data*"))]
    output_dir = Path(output_dir)
    output_dir.mkdir()
    for chunk_index, input_file in enumerate(files):
        filename = output_dir / get_collection_filename(chunk_index)
        output_file = BinaryFileWrite(filename)
        for item in shuffle_func(input_file):
            output_file.write(item)
        output_file.close()


def shuffle_each_collection_file(input_dir, output_dir):
    _shuffle_each_collection_file(input_dir, output_dir, shuffle_file)


def sort_each_collection_file(input_dir, output_dir, schema, key):
    shuffle_file = partial(sort_file, schema=schema, key=key)
    _shuffle_each_collection_file(input_dir, output_dir, shuffle_file)


def random_read_stream(files):
    random = RandomInt(len(files))
    while files:
        file_index = random.get_val()
        try:
            item = next(files[file_index])
            yield item
        except StopIteration:
            del files[file_index]
            if files:
                random.reset(n=len(files))
            continue


@total_ordering
class SortItem:
    def __init__(self, val, key, index):
        self.val = val
        self.key = key
        self.index = index

    def __eq__(self, other):
        return self.key == other.key

    def __lt__(self, other):
        return self.key < other.key


def sorted_read_stream(files, schema, key):
    h = []
    for index, i in enumerate(files):
        val = next(i)
        item = SortItem(val, key(schema.deserialize(val)), index)
        h.append(item)

    heapq.heapify(h)
    while h:
        item = heapq.heappop(h)
        yield item.val
        try:
            val = next(files[item.index])
            item = SortItem(val, key(schema.deserialize(val)), item.index)
            heapq.heappush(h, item)
        except StopIteration:
            pass


def _shuffle_collection(input_dir, output_dir, chunk_size, stream):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    files = [iter(BinaryFileRead(i).read()) for i in sorted(input_dir.glob("data*"))]
    output_file = BinaryFileCollectionWrite(output_dir, chunk_size)
    for item in stream(files):
        output_file.write(item)

    output_file.close()


def shuffle_collection(input_dir, output_dir, chunk_size):
    _shuffle_collection(input_dir, output_dir, chunk_size, random_read_stream)


def sort_collection(input_dir, output_dir, chunk_size, schema, key):
    stream = partial(sorted_read_stream, schema=schema, key=key)
    _shuffle_collection(input_dir, output_dir, chunk_size, stream)


def split_dataset(input_dir, dirs, fractions, chunk_size):
    if len(dirs) != len(fractions):
        raise ValueError(
            f"Number of dirs and fractions are different: {len(dirs)} vs. {len(fractions)}."
        )
    _, n_samples = dataset_statistics(input_dir)
    n_samples = sum(n_samples)
    target_samples = [round(n_samples * i) for i in fractions]
    diff = n_samples - sum(target_samples)
    target_samples[-1] += diff
    target_samples = [0] + target_samples
    target_samples = np.cumsum(target_samples)
    read_file = BinaryFileCollectionRead(input_dir).read()

    read_iterator = iter(read_file)
    for output_dir, start_idx, stop_idx in zip(
        dirs, target_samples[:-1], target_samples[1:]
    ):
        write_file = BinaryFileCollectionWrite(output_dir, chunk_size)
        for i in range(start_idx, stop_idx):
            item = next(read_iterator)
            write_file.write(item)
        write_file.close()
    read_file.close()


def sample_dataset(input_dir, output_dir, chunk_size, n_samples):
    """Routine to extract first n_samples from dataset."""
    read_file = BinaryFileCollectionRead(input_dir).read()
    write_file = BinaryFileCollectionWrite(output_dir, chunk_size)
    for index, item in enumerate(read_file):
        if index == n_samples:
            break
        write_file.write(item)
    write_file.close()
