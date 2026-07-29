"""The corpus/vector helpers used by the covered models."""

from __future__ import annotations

import numpy as np


def sparse2full(doc, length: int) -> np.ndarray:
    length = int(length)
    if length < 0:
        raise ValueError("length must be non-negative")
    result = np.zeros(length, dtype=np.float32)
    for index, value in doc:
        index = int(index)
        if index < 0 or index >= length:
            raise ValueError("term index is outside the requested vector length")
        result[index] = value
    return result


def full2sparse(vec, eps: float = 1e-9):
    values = np.asarray(vec)
    return [(int(i), float(values[i])) for i in np.flatnonzero(np.abs(values) > eps)]


def unitvec(vec, norm: str = "l2", return_norm: bool = False):
    if isinstance(vec, np.ndarray):
        result = np.asarray(vec)
        if norm == "l1":
            length = float(np.abs(result).sum())
        elif norm == "unique":
            length = float(np.count_nonzero(result))
        else:
            length = float(np.sqrt(np.dot(result, result)))
        normalized = result / length if length else result
    else:
        values = list(vec)
        if norm == "l1":
            length = sum(abs(value) for _, value in values)
        elif norm == "unique":
            length = float(len(values))
        else:
            length = float(np.sqrt(sum(value * value for _, value in values)))
        normalized = (
            [(index, value / length) for index, value in values] if length else values
        )
    return (normalized, length) if return_norm else normalized


def corpus2dense(corpus, num_terms: int, num_docs: int | None = None, dtype=np.float32):
    docs = list(corpus)
    if num_docs is not None and len(docs) != num_docs:
        raise ValueError("num_docs does not match corpus length")
    result = np.zeros((int(num_terms), len(docs)), dtype=dtype)
    for col, doc in enumerate(docs):
        for row, value in doc:
            result[int(row), col] = value
    return result


def dense2vec(vec, eps: float = 1e-9):
    return full2sparse(vec, eps)


def argsort(x, topn=None, reverse: bool = False):
    order = np.argsort(np.asarray(x))
    if reverse:
        order = order[::-1]
    return order if topn is None else order[:topn]
