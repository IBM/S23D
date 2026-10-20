import json
from collections import OrderedDict

import msgpack
import msgpack_numpy
import numpy as np

_B8 = 8


ITEM_SIZE_BYTES = _B8


def serialize_size(val):
    return np.int64(val).tobytes()


def deserialize_size(data):
    return np.frombuffer(data, dtype=np.int64).item()


class BinaryListSerializer:
    @staticmethod
    def serialize(data: list[bytes]):
        result = []
        for i in data:
            result.append(np.int64(len(i)).tobytes())
            result.append(i)
        return b"".join(result)

    @staticmethod
    def deserialize(data: bytes):
        n = len(data)
        index = 0
        result = []
        while index < n:
            size = np.frombuffer(data[index : index + _B8], dtype=np.uint64).item()
            index += _B8
            val = data[index : index + size]
            result.append(val)
            index += size
        return result


def dict_to_ordered_dict(d):
    od = OrderedDict([(key, d[key]) for key in sorted(d)])
    return od


class DictionarySchema:
    def __init__(self, schema):
        self.schema = dict_to_ordered_dict(schema)

    def serialize(self, data: dict):
        vals = []
        for key, value in self.schema.items():
            vals.append(value.serialize(data[key]))
        return BinaryListSerializer.serialize(vals)

    def deserialize(self, data: bytes):
        bin_vals = BinaryListSerializer.deserialize(data)
        result = {}
        for v, (key, serializer) in zip(bin_vals, self.schema.items()):
            result[key] = serializer.deserialize(v)
        return result

    def to_dict(self):
        obj = [
            self.__class__.__name__,
            [[key, value.to_dict()] for key, value in self.schema.items()],
        ]
        return obj

    @staticmethod
    def from_dict(data):
        schema = OrderedDict(
            (key, SchemaFactory.build_schema(value)) for key, value in data
        )
        return DictionarySchema(schema)

    def __setitem__(self, idx, value):
        self.schema[idx] = value
        self.schema = dict_to_ordered_dict(self.schema)

    def __getitem__(self, idx):
        return self.schema[idx]

    def __delitem__(self, key):
        del self.schema[key]

    def copy(self):
        return DictionarySchema(self.schema.copy())


class ListSchema:
    def __init__(self, schema):
        self.schema = schema

    def serialize(self, data):
        items = []
        for i in data:
            items.append(self.schema.serialize(i))
        return BinaryListSerializer.serialize(items)

    def deserialize(self, data):
        data = BinaryListSerializer.deserialize(data)
        items = []
        for i in data:
            items.append(self.schema.deserialize(i))
        return items

    def to_dict(self):
        obj = [self.__class__.__name__, self.schema.to_dict()]
        return obj

    @staticmethod
    def from_dict(data):
        schema = SchemaFactory.build_schema(data)
        return ListSchema(schema)


class StrSchema:
    @staticmethod
    def serialize(data: str):
        return data.encode()

    @staticmethod
    def deserialize(data: bytes):
        return data.decode()

    @staticmethod
    def to_dict():
        return [StrSchema.__name__, None]

    @staticmethod
    def from_dict(data):
        return StrSchema


class NumpySchema:
    @staticmethod
    def serialize(data):
        return msgpack.packb(data, default=msgpack_numpy.encode)

    @staticmethod
    def deserialize(data):
        return msgpack.unpackb(data, object_hook=msgpack_numpy.decode)

    @staticmethod
    def to_dict():
        return [NumpySchema.__name__, None]

    @staticmethod
    def from_dict(data):
        return NumpySchema


class ObjectSchema:
    @staticmethod
    def serialize(data):
        return msgpack.packb(data)

    def deserialize(data):
        return msgpack.unpackb(data)

    @staticmethod
    def to_dict():
        return [ObjectSchema.__name__, None]

    @staticmethod
    def from_dict(data):
        return ObjectSchema


class SchemaFactory:
    libs = {
        DictionarySchema.__name__: DictionarySchema,
        ObjectSchema.__name__: ObjectSchema,
        StrSchema.__name__: StrSchema,
        ListSchema.__name__: ListSchema,
        NumpySchema.__name__: NumpySchema,
    }

    @staticmethod
    def build_schema(data: dict):
        return SchemaFactory.libs[data[0]].from_dict(data[1])

    @staticmethod
    def str_to_schema(s: str):
        return SchemaFactory.build_schema(json.loads(s))

    @staticmethod
    def schema_to_str(schema):
        return json.dumps(schema.to_dict())
