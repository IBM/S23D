from itertools import islice

import cramjam
import msgpack
import msgpack_numpy
import numpy as np


def compress_index_list(index: list):
    """
    Sequence compression.
    Conversion of an index list to a list where a run of k > 1 consecutive
    indices a, a + 1, ..., a + k - 1 is written as the pair a, -k, and an index
    that is not part of a run is written as itself.
    """
    compressed_index = []
    first = index[0]
    prev_index = index[0]
    for i in islice(index, 1, None):
        if i != prev_index + 1:
            if first == prev_index:
                compressed_index.append(first)
            else:
                compressed_index.extend([first, -(prev_index - first + 1)])
            first = i
        prev_index = i

    if first == prev_index:
        compressed_index.append(first)
    else:
        compressed_index.extend([first, -(prev_index - first + 1)])
    return compressed_index


def decompress_index_list(compressed_index: list):
    """
    Sequence decompression.
    Inverse of `compress_index_list`: a non-negative value is an index, and a
    negative value -k after index a expands to the indices a + 1, ..., a + k - 1.
    """
    index = []
    prev_index = 0
    for i in compressed_index:
        if i >= 0:
            index.append(i)
        else:
            index.extend(list(range(prev_index + 1, prev_index - i)))
        prev_index = i

    return index


class IndexSerializer:
    @staticmethod
    def serialize(index: list[int]):
        index = compress_index_list(index)
        index = np.array(index, dtype=np.int64)
        return cramjam.snappy.compress(
            msgpack.packb(index, default=msgpack_numpy.encode)
        )

    @staticmethod
    def deserialize(data: bytes):
        index = msgpack.unpackb(
            cramjam.snappy.decompress(data), object_hook=msgpack_numpy.decode
        )
        return decompress_index_list(index)
