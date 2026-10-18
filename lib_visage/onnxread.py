"""Read initializers straight from an .onnx file's protobuf wire format.

The `onnx` package needs a matching `protobuf`, and Forge installs do not always have one
(onnx 1.23 with an older protobuf fails on `google.protobuf.runtime_version`). Visage only
needs one tensor (the inswapper emap), so it walks the bytes itself.
"""

from __future__ import annotations

import mmap

import numpy as np

# TensorProto.DataType -> numpy
_DTYPES = {1: np.float32, 10: np.float16, 11: np.float64}


def _varint(buf, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _fields(buf, start: int, end: int):
    """Yield (field number, wire type, value) where value is an int or a (start, end) span."""
    pos = start
    while pos < end:
        key, pos = _varint(buf, pos)
        num, wire = key >> 3, key & 7
        if wire == 0:
            val, pos = _varint(buf, pos)
        elif wire == 1:
            val, pos = (pos, pos + 8), pos + 8
        elif wire == 2:
            n, pos = _varint(buf, pos)
            val, pos = (pos, pos + n), pos + n
        elif wire == 5:
            val, pos = (pos, pos + 4), pos + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield num, wire, val


def _packed_varints(buf, span) -> list[int]:
    out, pos = [], span[0]
    while pos < span[1]:
        v, pos = _varint(buf, pos)
        out.append(v)
    return out


def _tensor(buf, span) -> np.ndarray:
    dims: list[int] = []
    dtype = 1
    raw = None
    floats: list[bytes] = []
    halves: list[int] = []
    for num, wire, val in _fields(buf, *span):
        if num == 1:  # dims
            dims.extend(_packed_varints(buf, val) if wire == 2 else [val])
        elif num == 2:  # data_type
            dtype = val
        elif num == 4:  # float_data
            floats.append(bytes(buf[val[0]:val[1]]))
        elif num == 5:  # int32_data, which also carries float16 bits
            halves.extend(_packed_varints(buf, val) if wire == 2 else [val])
        elif num == 9:  # raw_data
            raw = bytes(buf[val[0]:val[1]])
        elif num == 14 and val == 1:  # data_location EXTERNAL
            raise ValueError("tensor data is stored outside the .onnx file")
    if dtype not in _DTYPES:
        raise ValueError(f"unsupported tensor data type {dtype}")
    np_dtype = _DTYPES[dtype]
    if raw is not None:
        arr = np.frombuffer(raw, dtype=np.dtype(np_dtype).newbyteorder("<"))
    elif dtype == 10:
        arr = np.array(halves, dtype=np.uint16).view(np.float16)
    else:
        arr = np.frombuffer(b"".join(floats), dtype=np.dtype(np_dtype).newbyteorder("<"))
    return arr.reshape(dims).astype(np_dtype)


def last_initializer(path: str) -> np.ndarray:
    """The graph's last initializer (ModelProto.graph = 7, GraphProto.initializer = 5)."""
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        graph = next((val for num, wire, val in _fields(buf, 0, len(buf)) if num == 7 and wire == 2), None)
        if graph is None:
            raise ValueError(f"{path}: no graph found")
        last = None
        for num, wire, val in _fields(buf, *graph):
            if num == 5 and wire == 2:
                last = val
        if last is None:
            raise ValueError(f"{path}: graph has no initializers")
        return _tensor(buf, last)
