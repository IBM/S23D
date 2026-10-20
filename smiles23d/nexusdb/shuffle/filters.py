from smiles23d.nexusdb.index import build_index

from .io import BinaryFileCollectionRead, BinaryFileCollectionWrite


def filter_dataset(input_dir, output_dir, schema, filter, chunk_size):
    input_file = BinaryFileCollectionRead(input_dir)
    output_file = BinaryFileCollectionWrite(output_dir, chunk_size)
    for binary_item in input_file.read():
        item = schema.deserialize(binary_item)
        if filter(item):
            output_file.write(binary_item)
    output_file.close()


def filter_dataset_to_index(
    input_dir, output_dir, schema, filter=None, chunk_size=10000
):
    input_file = BinaryFileCollectionRead(input_dir)

    def index_filter_stream():
        for index, binary_item in enumerate(input_file.read()):
            item = schema.deserialize(binary_item)
            for i in range(int(filter(item))):
                yield index

    def index_stream():
        for index, binary_item in enumerate(input_file.read()):
            yield index

    if filter is not None:
        build_index(output_dir, index_filter_stream(), schema, chunk_size)
    else:
        build_index(output_dir, index_stream(), schema, chunk_size)
