import os
from itertools import islice
from typing import Any

import numcodecs
import numpy as np
import zarr
from litdata import StreamingDataLoader, StreamingDataset
from litdata.streaming.item_loader import BaseItemLoader, Interval

from smiles23d.nexusdb.datasets.nexus import OptionalStateLoadMixin
from smiles23d.nexusdb.index.serializers import IndexSerializer
from smiles23d.nexusdb.shuffle.io import BinaryFileCollectionRead, dataset_statistics
from smiles23d.nexusdb.shuffle.serialization import SchemaFactory


def batched(iterable, n):
    """From itertools 3.12."""
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


def build_zarr(input_dir, output_dir, chunk_size, compressor=None):
    compressor = (
        compressor
        if compressor
        else numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
    )
    _, n_samples = dataset_statistics(input_dir)
    n_samples = sum(n_samples)
    zarr_file = zarr.open_array(
        output_dir,
        mode="w",
        shape=(n_samples,),
        chunks=(chunk_size,),
        dtype=object,
        object_codec=numcodecs.VLenBytes(),
        compressor=compressor,
        zarr_version=2,
    )

    offset = 0
    for items in batched(BinaryFileCollectionRead(input_dir).read(), n=1024):
        item_size = len(items)
        zarr_file[offset : offset + item_size] = np.array(items)
        offset += item_size


def read_zarr(input_dir, buffer_size=1024):
    array = zarr.open(input_dir, mode="r")
    n = int(len(array) // buffer_size) * buffer_size + buffer_size
    if n <= buffer_size:
        yield from array[:]
    for start_idx, end_idx in zip(
        range(0, n - buffer_size, buffer_size), range(buffer_size, n, buffer_size)
    ):
        yield from array[start_idx:end_idx]


class NexusZarrItemLoader(BaseItemLoader):
    """The base item loader is responsible to decide how the items within a chunk are loaded."""

    def __init__(self, dataset_dir, index_dir, start_idx, end_idx, *args, **kargs):
        self.dataset_dir = dataset_dir
        self.index_dir = index_dir
        self.start_idx, self.end_idx = start_idx, end_idx

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)

        nexus_opt = self._config["nexus"]
        self.schema = SchemaFactory.build_schema(nexus_opt["schema"])

        self.chunk_size = self._config["chunk_size"]
        self.buffer = None
        self.buffer_chunk_index = None

    def state_dict(self) -> dict:
        return {}

    def generate_intervals(self) -> list[tuple[int, int]]:
        """Returns a list of tuple describing the indexes intervals of the chunks."""
        intervals = []
        offset = 0
        for chunk in self._chunks:
            chunk_size = chunk["chunk_size"]
            start_idx, end_idx = offset, offset + chunk_size
            roi_start, roi_end = start_idx, end_idx
            if self.start_idx:
                roi_start = min(max(self.start_idx, start_idx), end_idx)
            if self.end_idx:
                roi_end = max(start_idx, min(self.end_idx, end_idx))
            intervals.append(Interval(start_idx, roi_start, roi_end, end_idx))
            offset += chunk_size
        return intervals

    def pre_load_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        """Logic to load the chunk in background to gain some time."""
        pass

    def load_item_from_chunk(
        self,
        index: int,
        chunk_index: int,
        chunk_filepath: str,
        begin: int,
        chunk_bytes: int,
    ) -> Any:
        if self.buffer is None:
            self.zarr_file = zarr.open(self.dataset_dir, mode="r")
        if self.buffer_chunk_index != chunk_index:
            with open(os.path.join(self.index_dir, chunk_filepath), "rb") as file:
                indices = IndexSerializer.deserialize(file.read())
                self.buffer = self.zarr_file[indices]
            self.buffer_chunk_index = chunk_index
        item_index = index - begin
        return self.schema.deserialize(self.buffer[item_index])

    def delete(self, chunk_index: int, chunk_filepath: str) -> None:
        """Delete a chunk from the local filesystem."""
        pass

    def encode_data(
        self, data: list[bytes], sizes: list[int], flattened: list[Any]
    ) -> Any:
        raise RuntimeError("Not supposed to call this method.")


class NexusZarrDataset(StreamingDataset):
    def __init__(self, dataset_dir, index_dir, start_idx=None, end_idx=None, **kargs):
        dataset_dir = str(dataset_dir)
        index_dir = str(index_dir)
        item_loader = NexusZarrItemLoader(dataset_dir, index_dir, start_idx, end_idx)
        super().__init__(index_dir, item_loader=item_loader, **kargs)


class NexusZarrDataloader(OptionalStateLoadMixin, StreamingDataLoader):
    pass
