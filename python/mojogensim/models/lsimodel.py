"""Randomized sparse LSI whose large matrix products run in Mojo."""

from __future__ import annotations

from dataclasses import dataclass
import pickle

import numpy as np

from .._lib import addr, f64, i64, lib


@dataclass
class Projection:
    u: np.ndarray
    s: np.ndarray


def _csr(corpus, num_terms):
    docs = [list(doc) for doc in corpus]
    indptr = [0]
    indices = []
    values = []
    for doc in docs:
        for index, value in doc:
            index = int(index)
            if index < 0 or index >= num_terms:
                raise ValueError("term id outside id2word/num_terms range")
            if value:
                indices.append(index)
                values.append(float(value))
        indptr.append(len(indices))
    return docs, i64(indptr), i64(indices), f64(values)


def _spmm(indptr, indices, values, right, rows, cols):
    right = f64(right)
    result = np.zeros((rows, right.shape[1]), dtype=np.float64)
    if rows and cols and len(values):
        lib().mg_csr_matmul(
            addr(indptr, np.int64),
            addr(indices, np.int64),
            addr(values, np.float64),
            addr(right, np.float64),
            addr(result, np.float64, writable=True),
            rows,
            cols,
            right.shape[1],
        )
    return result


def _spmm_t(indptr, indices, values, right, rows, cols):
    right = f64(right)
    result = np.zeros((cols, right.shape[1]), dtype=np.float64)
    if rows and cols and len(values):
        lib().mg_csr_t_matmul(
            addr(indptr, np.int64),
            addr(indices, np.int64),
            addr(values, np.float64),
            addr(right, np.float64),
            addr(result, np.float64, writable=True),
            rows,
            cols,
            right.shape[1],
        )
    return result


class LsiModel:
    def __init__(
        self,
        corpus=None,
        num_topics=200,
        id2word=None,
        chunksize=20000,
        decay=1.0,
        distributed=False,
        onepass=True,
        power_iters=2,
        extra_samples=100,
        dtype=np.float64,
        random_seed=None,
        **kwargs,
    ):
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported LsiModel arguments: {names}")
        if distributed:
            raise NotImplementedError("distributed LSI is not covered")
        self.num_topics = int(num_topics)
        if self.num_topics <= 0:
            raise ValueError("num_topics must be positive")
        self.id2word = id2word
        self.chunksize = int(chunksize)
        self.decay = float(decay)
        self.onepass = bool(onepass)
        self.power_iters = int(power_iters)
        self.extra_samples = int(extra_samples)
        if self.power_iters < 0 or self.extra_samples < 0:
            raise ValueError("power_iters and extra_samples must be non-negative")
        self.dtype = np.dtype(dtype)
        if self.dtype.kind != "f":
            raise TypeError("LSI dtype must be floating point")
        self.random_seed = random_seed
        self.projection = Projection(
            np.empty((0, self.num_topics), dtype=self.dtype),
            np.empty(0, dtype=self.dtype),
        )
        self.docs_processed = 0
        self._corpus = []
        if corpus is not None:
            self.add_documents(corpus)

    def _num_terms(self, docs):
        if self.id2word is not None:
            keys = list(self.id2word.keys()) if hasattr(self.id2word, "keys") else range(len(self.id2word))
            return max(keys, default=-1) + 1
        return max((index for doc in docs for index, _ in doc), default=-1) + 1

    def add_documents(self, corpus, chunksize=None, decay=None):
        new_docs = [list(doc) for doc in corpus]
        self._corpus.extend(new_docs)
        self.docs_processed = len(self._corpus)
        terms = self._num_terms(self._corpus)
        if not self._corpus or terms == 0:
            return
        docs, indptr, indices, values = _csr(self._corpus, terms)
        rows = len(docs)
        rank = min(self.num_topics, rows, terms)
        sketch_width = min(
            rows,
            terms,
            rank + max(0, self.extra_samples),
        )
        rng = np.random.default_rng(self.random_seed)
        omega = rng.standard_normal((terms, sketch_width))
        y = _spmm(indptr, indices, values, omega, rows, terms)
        for _ in range(self.power_iters):
            q, _ = np.linalg.qr(y, mode="reduced")
            z = _spmm_t(indptr, indices, values, q, rows, terms)
            y = _spmm(indptr, indices, values, z, rows, terms)
        q, _ = np.linalg.qr(y, mode="reduced")
        b = _spmm_t(indptr, indices, values, q, rows, terms).T
        _, singular, vt = np.linalg.svd(b, full_matrices=False)
        u = vt[:rank].T
        for col in range(rank):
            pivot = np.argmax(np.abs(u[:, col]))
            if u[pivot, col] < 0:
                u[:, col] *= -1
        self.projection = Projection(
            np.asarray(u, dtype=self.dtype),
            np.asarray(singular[:rank], dtype=self.dtype),
        )
        self.num_topics = rank

    def _transform_doc(self, bow, scaled=False):
        result = np.zeros(self.num_topics, dtype=self.dtype)
        for index, value in bow:
            index = int(index)
            if 0 <= index < self.projection.u.shape[0]:
                result += float(value) * self.projection.u[index]
        if scaled and len(self.projection.s):
            result /= np.where(self.projection.s == 0, 1.0, self.projection.s)
        return [
            (topic, float(value))
            for topic, value in enumerate(result)
            if abs(value) > 1e-12
        ]

    def __getitem__(self, bow, scaled=False, chunksize=None):
        if isinstance(bow, np.ndarray):
            if bow.ndim == 1:
                bow = list(enumerate(bow))
            else:
                return [self._transform_doc(list(enumerate(row)), scaled) for row in bow]
        materialized = list(bow)
        if not materialized:
            return []
        if isinstance(materialized[0], tuple):
            return self._transform_doc(materialized, scaled)
        return [self._transform_doc(doc, scaled) for doc in materialized]

    def get_topics(self):
        return self.projection.u.T.copy()

    def show_topic(self, topicno, topn=10):
        weights = self.projection.u[:, int(topicno)]
        order = np.argsort(-np.abs(weights))[:topn]
        def token(index):
            if self.id2word is None:
                return str(index)
            return self.id2word[index]
        return [(token(int(index)), float(weights[index])) for index in order]

    def show_topics(self, num_topics=-1, num_words=10, log=False, formatted=True):
        count = self.num_topics if num_topics < 0 else min(num_topics, self.num_topics)
        result = []
        for topic in range(count):
            terms = self.show_topic(topic, num_words)
            value = " + ".join(f'{weight:.3f}*"{word}"' for word, weight in terms) if formatted else terms
            result.append((topic, value))
        return result

    def print_topic(self, topicno, topn=10):
        return " + ".join(
            f'{weight:.3f}*"{word}"'
            for word, weight in self.show_topic(topicno, topn)
        )

    def save(self, fname, **kwargs):
        with open(fname, "wb") as stream:
            pickle.dump(self, stream)

    @classmethod
    def load(cls, fname, **kwargs):
        with open(fname, "rb") as stream:
            return pickle.load(stream)
