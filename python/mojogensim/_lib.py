"""ctypes bridge to the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-gensim.so")

I = ctypes.c_int64
F32 = ctypes.c_float

_SIGNATURES = {
    "mg_normalize_rows": ([I] * 4, None),
    "mg_dot_rows": ([I] * 5, None),
    "mg_cosine_rows": ([I] * 6, None),
    "mg_sg_pairs": ([I] * 7, I),
    "mg_sgns_train": ([I] * 8 + [F32], None),
    "mg_cbow_train": ([I] * 11 + [F32, I], None),
    "mg_dm_train": ([I] * 13 + [F32, I, I], None),
    "mg_csr_matmul": ([I] * 8, None),
    "mg_csr_t_matmul": ([I] * 8, None),
}

_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB):
            raise RuntimeError("Mojo library not built; run `pixi run build`")
        _library = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(
    array: np.ndarray,
    dtype,
    *,
    writable: bool = False,
) -> int:
    """Return an address only when a buffer exactly matches its Mojo pointer type."""
    if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
        raise TypeError("FFI buffers must be C-contiguous NumPy arrays")
    expected = np.dtype(dtype)
    if array.dtype != expected:
        raise TypeError(
            f"FFI buffer has dtype {array.dtype}; expected exactly {expected}"
        )
    if array.size == 0:
        raise ValueError("empty buffers must not cross the Mojo FFI")
    if writable and not array.flags.writeable:
        raise TypeError("FFI output buffers must be writable")
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("null buffers must not cross the Mojo FFI")
    return address


def f32(value, *, copy: bool = False) -> np.ndarray:
    source = np.asarray(value)
    if np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError("complex values cannot be narrowed to float32")
    with np.errstate(over="ignore", invalid="ignore"):
        result = (
            np.array(source, dtype=np.float32, order="C", copy=True)
            if copy
            else np.ascontiguousarray(source, dtype=np.float32)
        )
    if np.isfinite(source).all() and not np.isfinite(result).all():
        raise OverflowError("value is outside the float32 range")
    return result


def f64(value, *, copy: bool = False) -> np.ndarray:
    source = np.asarray(value)
    if np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError("complex values cannot be narrowed to float64")
    return (
        np.array(source, dtype=np.float64, order="C", copy=True)
        if copy
        else np.ascontiguousarray(source, dtype=np.float64)
    )


def i64(value) -> np.ndarray:
    source = np.asarray(value)
    if not (
        np.issubdtype(source.dtype, np.integer)
        or np.issubdtype(source.dtype, np.bool_)
    ):
        raise TypeError("integer FFI buffers require integer values")
    if np.issubdtype(source.dtype, np.unsignedinteger) and source.size:
        if int(source.max()) > np.iinfo(np.int64).max:
            raise OverflowError("integer is outside the int64 range")
    return np.ascontiguousarray(source, dtype=np.int64)


def normalize_rows(vectors) -> np.ndarray:
    source = f32(vectors)
    if source.ndim != 2:
        raise ValueError("expected a 2D matrix")
    result = np.empty_like(source)
    if source.size:
        lib().mg_normalize_rows(
            addr(source, np.float32),
            addr(result, np.float32, writable=True),
            source.shape[0],
            source.shape[1],
        )
    return result


def dot_rows(matrix, query) -> np.ndarray:
    matrix = f32(matrix)
    query = f32(query)
    if matrix.ndim != 2 or query.shape != (matrix.shape[1],):
        raise ValueError("matrix/query dimensions do not agree")
    result = np.empty(matrix.shape[0], dtype=np.float32)
    if matrix.size:
        lib().mg_dot_rows(
            addr(matrix, np.float32),
            addr(query, np.float32),
            addr(result, np.float32, writable=True),
            matrix.shape[0],
            matrix.shape[1],
        )
    return result


def cosine_rows(matrix, norms, query) -> np.ndarray:
    matrix = f32(matrix)
    norms = f32(norms)
    query = f32(query)
    if (
        matrix.ndim != 2
        or norms.shape != (matrix.shape[0],)
        or query.shape != (matrix.shape[1],)
    ):
        raise ValueError("matrix/norm/query dimensions do not agree")
    result = np.empty(matrix.shape[0], dtype=np.float32)
    if matrix.size:
        lib().mg_cosine_rows(
            addr(matrix, np.float32),
            addr(norms, np.float32),
            addr(query, np.float32),
            addr(result, np.float32, writable=True),
            matrix.shape[0],
            matrix.shape[1],
        )
    return result
