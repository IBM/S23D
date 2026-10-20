import multiprocessing as mp
import os
import shutil
import uuid
from collections.abc import Iterable
from itertools import islice
from pathlib import Path

from .io import (
    BinaryFileCollectionRead,
    BinaryFileCollectionWrite,
    get_collection_filename,
    get_collection_glob_string,
)


def build_dataset(data_generator, schema, output_dir, chunk_size):
    file = BinaryFileCollectionWrite(output_dir, chunk_size)
    for item in data_generator:
        file.write(schema.serialize(item))
    file.close()


def read_dataset(input_dir, schema):
    file = BinaryFileCollectionRead(input_dir)
    for binary_item in file.read():
        item = schema.deserialize(binary_item)
        yield item


def combine_datasets(input_dirs, output_dir):
    input_dirs = [Path(input_dir) for input_dir in input_dirs]
    output_dir = Path(output_dir)
    files = []
    for input_dir in input_dirs:
        files.extend(sorted(input_dir.glob("data*")))

    output_dir.mkdir()
    for index, i in enumerate(files):
        shutil.move(i, output_dir / get_collection_filename(index))

    for input_dir in input_dirs:
        os.rmdir(str(input_dir))


def _batched(iterable, n):
    """From itertools for compatibility with old python."""
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


def _worker(data_generator, output_dir, schema, queue, chunk_size):
    output_dir = Path(output_dir) / ("datafolder" + str(uuid.uuid4()))
    file = BinaryFileCollectionWrite(str(output_dir), chunk_size)
    chunk = queue.get()
    while chunk:
        for chunk_item in chunk:
            for item in data_generator(chunk_item):
                file.write(schema.serialize(item))
        chunk = queue.get()
    file.close()


def _fan_data_to_processes(data: Iterable, n_process, chunk_size):
    process_data = None
    for row_index, row in enumerate(_batched(data, n_process)):
        if row_index % chunk_size == 0:
            if process_data:
                yield process_data
            process_data = [[] for i in range(n_process)]
        for index, j in enumerate(row):
            process_data[index].append(j)

    if process_data[0]:
        yield process_data


def build_dataset_multiprocess(
    data_generator,
    generator_inputs,
    schema,
    output_dir,
    chunk_size,
    n_processes,
    queue_chunk_size,
):
    output_dir = Path(output_dir)
    output_dir.mkdir()

    processes, queues = [], []
    for _ in range(n_processes):
        queue = mp.Queue()
        process = mp.Process(
            target=_worker,
            args=(data_generator, output_dir, schema, queue, chunk_size),
        )
        queues.append(queue)
        processes.append(process)
        process.start()

    for data in _fan_data_to_processes(generator_inputs, n_processes, queue_chunk_size):
        for d, queue in zip(data, queues):
            if d:
                queue.put(d)
    for queue in queues:
        queue.put(None)

    for process in processes:
        process.join()

    folders = output_dir.glob("datafolder*")
    glob_string = get_collection_glob_string()
    file_index = 0
    for folder in folders:
        files = folder.glob(glob_string)
        for file in files:
            output_filename = output_dir / get_collection_filename(file_index)
            file_index += 1
            shutil.move(file, output_filename)
        shutil.rmtree(folder)
