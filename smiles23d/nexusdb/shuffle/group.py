from .io import BinaryFileCollectionRead, BinaryFileCollectionWrite


def unroll_schema(
    key: str, new_key: str, schema: dict, input_dir, output_dir, chunk_size
):
    unroll_schema = schema.copy()
    unroll_schema[new_key] = unroll_schema[key].schema
    del unroll_schema[key]
    output_file = BinaryFileCollectionWrite(output_dir, chunk_size)

    for binary_item in BinaryFileCollectionRead(input_dir).read():
        item = schema.deserialize(binary_item)
        unroll_item = item[key]
        del item[key]
        for i in unroll_item:
            item[new_key] = i
            output_file.write(unroll_schema.serialize(item))

    output_file.close()


def group_dataset(
    key: str,
    new_key: str,
    schema: dict,
    new_schema: dict,
    input_dir,
    output_dir,
    chunk_size,
):
    read_file = BinaryFileCollectionRead(input_dir).read()
    output_file = BinaryFileCollectionWrite(output_dir, chunk_size)
    prev_key = None

    group = []
    for binary_item in read_file:
        item = schema.deserialize(binary_item)
        item_key = item[key]
        del item[key]
        if prev_key != item_key:
            if prev_key:
                obj = {key: prev_key, new_key: group}
                new_item = new_schema.serialize(obj)
                output_file.write(new_item)
            group = []
        group.append(item)
        prev_key = item_key

    if group:
        obj = {key: prev_key, new_key: group}
        item = new_schema.serialize(obj)
        output_file.write(item)
    output_file.close()
