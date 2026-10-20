import json
import os
from typing import Any

from litdata import StreamingDataLoader, StreamingDataset
from litdata.streaming.item_loader import BaseItemLoader, Interval

from smiles23d.nexusdb.shuffle.io import BinaryFileRead, dataset_statistics
from smiles23d.nexusdb.shuffle.serialization import SchemaFactory
from smiles23d.nexusdb.shuffle.shuffle import sample_dataset


def copy_schema_and_build_index(input_dir, output_dir):
    with open(os.path.join(input_dir, "index.json")) as file:
        obj = json.load(file)
    nexus_opt = obj["config"]["nexus"]
    schema = SchemaFactory.build_schema(nexus_opt["schema"])
    build_litdata_dataset_index(output_dir, schema)


def build_litdata_dataset_index(input_dir, schema, ctx=None):
    """The function that adds index.json that is required by LitData."""
    files, n_samples = dataset_statistics(input_dir)

    chunks = []
    for filename, sample_size in zip(files, n_samples):
        chunk = {
            "chunk_bytes": 1,
            "chunk_size": sample_size,
            "dim": None,
            "filename": filename,
        }
        chunks.append(chunk)

    config = {
        "chunk_bytes": None,
        "chunk_size": n_samples[0],
        "compression": None,
        "data_format": ["bytes"],
        "data_spec": '[1, {"type": null, "context": null, "children_spec": []}]',
        "nexus": {"schema": schema.to_dict()},
    }

    specs = {"chunks": chunks, "config": config}

    with open(os.path.join(input_dir, "index.json"), "w") as file:
        json.dump(specs, file)


class NexusItemLoader(BaseItemLoader):
    """The base item loader is responsible to decide how the items within a chunk are loaded."""

    def __init__(self, input_dir, start_idx, end_idx, *args, **kargs):
        self.input_dir = input_dir
        self.start_idx = start_idx
        self.end_idx = end_idx

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
        if self.buffer_chunk_index != chunk_index:
            self.buffer = list(
                BinaryFileRead(os.path.join(self.input_dir, chunk_filepath)).read()
            )
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


class NexusDataset(StreamingDataset):
    def __init__(self, input_dir, start_idx=None, end_idx=None, **kargs):
        input_dir = str(input_dir)
        item_loader = NexusItemLoader(input_dir, start_idx, end_idx)
        self.start_idx = start_idx
        super().__init__(input_dir, item_loader=item_loader, **kargs)


class OptionalStateLoadMixin:
    def __init__(self, *args, load_state: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_state = load_state

    def load_state_dict(self, obj):
        if not self.load_state:
            return
        super().load_state_dict(obj)


class NexusDataloader(OptionalStateLoadMixin, StreamingDataLoader):
    pass


def sample_nexus_dataset(input_dir, output_dir, chunk_size, n_samples):
    sample_dataset(input_dir, output_dir, chunk_size, n_samples)
    config_file = os.path.join(input_dir, "index.json")
    with open(config_file) as file:
        obj = json.load(file)

    schema = SchemaFactory.build_schema(obj["config"]["nexus"]["schema"])
    build_litdata_dataset_index(output_dir, schema)
