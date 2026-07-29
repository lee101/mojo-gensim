"""Exact cosine indexes with Mojo row scoring."""

from __future__ import annotations

import os
import pickle

import numpy as np

from .._lib import dot_rows, f32, normalize_rows
from ..matutils import sparse2full


def _is_bow(value):
    return isinstance(value, (list, tuple)) and (
        not value or isinstance(value[0], tuple)
    )


class MatrixSimilarity:
    def __init__(
        self,
        corpus,
        num_best=None,
        num_features=None,
        chunksize=256,
        dtype=np.float32,
        **kwargs,
    ):
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported MatrixSimilarity arguments: {names}")
        docs = list(corpus)
        self.num_best = num_best
        self.chunksize = int(chunksize)
        if num_features is None:
            if not docs:
                raise ValueError("cannot infer num_features from an empty corpus")
            if _is_bow(docs[0]):
                num_features = max(
                    (index for doc in docs for index, _ in doc), default=-1
                ) + 1
            else:
                num_features = np.asarray(docs[0]).size
        self.num_features = int(num_features)
        if self.num_features < 0:
            raise ValueError("num_features must be non-negative")
        if np.dtype(dtype) != np.dtype(np.float32):
            raise TypeError("Mojo similarity indexes support dtype=float32 only")
        dense = (
            np.vstack([self._dense(doc) for doc in docs]).astype(dtype, copy=False)
            if docs
            else np.empty((0, self.num_features), dtype=dtype)
        )
        self.index = normalize_rows(dense)

    def __len__(self):
        return len(self.index)

    def _dense(self, query):
        if _is_bow(query):
            return sparse2full(query, self.num_features)
        value = f32(query)
        if value.shape != (self.num_features,):
            raise ValueError("query has the wrong number of features")
        return value

    def get_similarities(self, query):
        vector = self._dense(query)
        length = np.linalg.norm(vector)
        if length:
            vector = vector / length
        return dot_rows(self.index, f32(vector))

    def __getitem__(self, query):
        if isinstance(query, np.ndarray) and query.ndim == 2:
            return [self._one(row) for row in query]
        materialized = list(query) if not isinstance(query, np.ndarray) else query
        if isinstance(materialized, list) and materialized and not isinstance(
            materialized[0], tuple
        ) and isinstance(materialized[0], (list, tuple, np.ndarray)):
            return [self._one(row) for row in materialized]
        return self._one(materialized)

    def _one(self, query):
        similarities = self.get_similarities(query)
        if self.num_best is None:
            return similarities
        order = np.argsort(-similarities, kind="stable")[: self.num_best]
        return [(int(index), float(similarities[index])) for index in order]

    def __iter__(self):
        for vector in self.index:
            yield self._one(vector)

    def save(self, fname, **kwargs):
        with open(fname, "wb") as stream:
            pickle.dump(self, stream)

    @classmethod
    def load(cls, fname, **kwargs):
        with open(fname, "rb") as stream:
            return pickle.load(stream)


class SparseMatrixSimilarity(MatrixSimilarity):
    """API-compatible exact index; storage is dense for the covered subset."""


class Similarity(MatrixSimilarity):
    def __init__(
        self,
        output_prefix,
        corpus,
        num_features,
        num_best=None,
        chunksize=256,
        shardsize=32768,
        norm="l2",
        **kwargs,
    ):
        self.output_prefix = os.fspath(output_prefix)
        self.shardsize = int(shardsize)
        self.norm = norm
        super().__init__(
            corpus,
            num_best=num_best,
            num_features=num_features,
            chunksize=chunksize,
            **kwargs,
        )

    def add_documents(self, corpus):
        extra = np.vstack([self._dense(doc) for doc in corpus])
        combined = np.vstack((self.index, normalize_rows(extra)))
        self.index = combined

    def destroy(self):
        return None


class WmdSimilarity:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("word mover distance is outside the covered subset")
