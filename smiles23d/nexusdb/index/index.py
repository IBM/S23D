import json
import os
from collections.abc import Iterable

from .serializers import IndexSerializer


def push_index(
    output_dir: str,
    filename: str,
    index: list,
):
    data = IndexSerializer.serialize(index)
    with open(os.path.join(output_dir, filename), "wb") as file:
        file.write(data)


def get_chunk_filename(file_index):
    return f"chunk_index{file_index:08}.bin"


def write_index(
    output_dir: str,
    index: Iterable,
    chunk_size: int,
):
    """
    Original litdata chunks have serialized items in files with bin extension.
    We save indices of the items instead to the bin files and read data from the database.
    The function creates bin files for litdata that have indices of items in the chunks.
    """
    os.makedirs(output_dir, exist_ok=False)

    files = []
    chunk_index = []
    file_index = 0
    for i in index:
        chunk_index.append(i)
        if len(chunk_index) == chunk_size:
            filename = get_chunk_filename(file_index)
            files.append((filename, chunk_size))
            push_index(output_dir, filename, index=chunk_index)
            file_index += 1
            chunk_index = []

    if chunk_index:
        filename = get_chunk_filename(file_index)
        files.append((filename, len(chunk_index)))
        push_index(output_dir, filename, index=chunk_index)

    return files


def build_litdata_index(output_dir: str, files, schema):
    chunks = []
    for filename, chunk_size in files:
        chunk = {
            "chunk_bytes": 1,
            "chunk_size": chunk_size,
            "dim": None,
            "filename": os.path.basename(filename),
        }
        chunks.append(chunk)
    config = {
        "chunk_bytes": None,
        "chunk_size": 1,
        "compression": None,
        "data_format": ["bytes"],
        "data_spec": '[1, {"type": null, "context": null, "children_spec": []}]',
        "nexus": {"schema": schema.to_dict()},
    }

    specs = {"chunks": chunks, "config": config}

    with open(os.path.join(output_dir, "index.json"), "w") as file:
        json.dump(specs, file)


def build_index(output_dir: str, index: Iterable, schema, chunk_size: int):
    files = write_index(output_dir, index, chunk_size)
    build_litdata_index(output_dir, files, schema)
