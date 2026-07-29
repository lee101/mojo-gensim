"""Gensim-compatible keyed vector storage and exact cosine queries."""

from __future__ import annotations

import pickle

import numpy as np

from .._lib import cosine_rows, dot_rows, f32, normalize_rows


def _top_order(scores, count):
    count = min(max(0, int(count)), len(scores))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count == len(scores):
        return np.argsort(-scores, kind="stable")
    partition = np.argpartition(-scores, count - 1)[:count]
    cutoff = np.min(scores[partition])
    better = np.flatnonzero(scores > cutoff)
    equal = np.flatnonzero(scores == cutoff)[: count - len(better)]
    candidates = np.concatenate((better, equal))
    return candidates[np.lexsort((candidates, -scores[candidates]))]


class KeyedVectors:
    def __init__(
        self,
        vector_size: int,
        count: int = 0,
        dtype=np.float32,
        mapfile_path=None,
    ):
        self.vector_size = int(vector_size)
        self.index_to_key: list = [None] * int(count)
        self.key_to_index: dict = {}
        self.vectors = np.zeros((int(count), self.vector_size), dtype=dtype)
        self.norms: np.ndarray | None = None
        self.expandos: dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self.index_to_key)

    def __contains__(self, key):
        return self.has_index_for(key)

    def __getitem__(self, key_or_keys):
        if isinstance(key_or_keys, (list, tuple, np.ndarray)):
            return np.vstack([self.get_vector(key) for key in key_or_keys])
        return self.get_vector(key_or_keys)

    def __setitem__(self, keys, weights):
        if isinstance(keys, (list, tuple, np.ndarray)):
            self.add_vectors(keys, weights, replace=True)
        else:
            self.add_vector(keys, weights)

    def get_index(self, key, default=None):
        if isinstance(key, (int, np.integer)) and 0 <= int(key) < len(self):
            return int(key)
        if key in self.key_to_index:
            return self.key_to_index[key]
        if default is not None:
            return default
        raise KeyError(f"Key {key!r} not present")

    def get_vector(self, key, norm: bool = False):
        index = self.get_index(key)
        if norm:
            self.fill_norms()
            length = self.norms[index]
            return self.vectors[index] / length if length else self.vectors[index]
        return self.vectors[index]

    def has_index_for(self, key) -> bool:
        return (
            isinstance(key, (int, np.integer)) and 0 <= int(key) < len(self)
        ) or key in self.key_to_index

    def add_vector(self, key, vector):
        return self.add_vectors([key], np.asarray(vector).reshape(1, -1))[0]

    def add_vectors(self, keys, weights, replace: bool = False):
        keys = list(keys)
        weights = f32(weights)
        if weights.shape != (len(keys), self.vector_size):
            raise ValueError("weights have the wrong shape")
        indices = []
        additions = []
        vacancies = [
            index for index, key in enumerate(self.index_to_key) if key is None
        ]
        for key, vector in zip(keys, weights):
            if key in self.key_to_index:
                index = self.key_to_index[key]
                if replace:
                    self.vectors[index] = vector
            elif vacancies:
                index = vacancies.pop(0)
                self.key_to_index[key] = index
                self.index_to_key[index] = key
                self.vectors[index] = vector
            else:
                index = len(self.index_to_key)
                self.key_to_index[key] = index
                self.index_to_key.append(key)
                additions.append(vector)
            indices.append(index)
        if additions:
            block = np.asarray(additions, dtype=np.float32)
            self.vectors = (
                block if self.vectors.size == 0 else np.vstack((self.vectors, block))
            )
            for name, values in self.expandos.items():
                self.expandos[name] = np.pad(values, (0, len(additions)))
        self.norms = None
        return indices

    def resize_vectors(self, seed: int = 0):
        missing = len(self.index_to_key) - len(self.vectors)
        if missing > 0:
            rng = np.random.default_rng(seed)
            extra = rng.uniform(
                -0.5 / self.vector_size,
                0.5 / self.vector_size,
                size=(missing, self.vector_size),
            ).astype(np.float32)
            self.vectors = np.vstack((self.vectors, extra))
        return self.vectors

    def fill_norms(self, force: bool = False):
        if force or self.norms is None:
            self.norms = np.linalg.norm(self.vectors, axis=1).astype(np.float32)

    def get_normed_vectors(self):
        if not len(self):
            return self.vectors.copy()
        return normalize_rows(self.vectors)

    def get_mean_vector(
        self,
        keys,
        weights=None,
        pre_normalize=True,
        post_normalize=False,
        ignore_missing=True,
    ):
        keys = list(keys)
        if weights is None:
            weights = np.ones(len(keys), dtype=np.float32)
        if len(weights) != len(keys):
            raise ValueError("weights and keys must have the same length")
        result = np.zeros(self.vector_size, dtype=np.float32)
        total_weight = 0.0
        for key, weight in zip(keys, weights):
            if isinstance(key, np.ndarray):
                vector = f32(key)
            elif key in self:
                vector = self.get_vector(key, norm=pre_normalize)
            elif ignore_missing:
                continue
            else:
                raise KeyError(f"Key {key!r} not present")
            result += float(weight) * vector
            total_weight += abs(float(weight))
        if total_weight:
            result /= total_weight
        if post_normalize:
            length = np.linalg.norm(result)
            if length:
                result /= length
        return result

    def _mean_query(self, positive, negative, pre_normalize=True):
        if positive is None:
            positive = []
        if isinstance(positive, (str, np.ndarray)) and not (
            isinstance(positive, np.ndarray) and positive.ndim == 2
        ):
            positive = [positive]
        if negative is None:
            negative = []
        if isinstance(negative, str):
            negative = [negative]
        weighted = []
        excluded = set()
        for items, sign in ((positive, 1.0), (negative, -1.0)):
            for item in items:
                weight = sign
                if isinstance(item, tuple):
                    item, weight = item
                if isinstance(item, np.ndarray):
                    vector = f32(item)
                else:
                    excluded.add(self.get_index(item))
                    vector = self.get_vector(item, norm=pre_normalize)
                weighted.append((float(weight), vector))
        if not weighted:
            raise ValueError("cannot compute similarity with no input")
        query = np.mean([weight * vector for weight, vector in weighted], axis=0)
        length = np.linalg.norm(query)
        if length:
            query = query / length
        return f32(query), excluded

    def most_similar(
        self,
        positive=None,
        negative=None,
        topn=10,
        clip_start=0,
        clip_end=None,
        restrict_vocab=None,
        indexer=None,
    ):
        if indexer is not None:
            return indexer.most_similar(positive, topn)
        query, excluded = self._mean_query(positive, negative)
        end = len(self) if clip_end is None else min(int(clip_end), len(self))
        if restrict_vocab is not None:
            end = min(end, int(restrict_vocab))
        start = int(clip_start)
        self.fill_norms()
        scores = cosine_rows(
            self.vectors[start:end],
            self.norms[start:end],
            query,
        )
        if topn is False:
            return scores
        requested = int(topn)
        if requested <= 0:
            return []
        order = _top_order(scores, requested + len(excluded))
        result = []
        for local in order:
            index = start + int(local)
            if index not in excluded:
                result.append((self.index_to_key[index], float(scores[local])))
                if len(result) >= int(topn):
                    break
        return result

    def similar_by_word(self, word, topn=10, restrict_vocab=None):
        return self.most_similar(
            positive=[word], topn=topn, restrict_vocab=restrict_vocab
        )

    similar_by_key = similar_by_word

    def similar_by_vector(self, vector, topn=10, restrict_vocab=None):
        return self.most_similar(
            positive=[vector], topn=topn, restrict_vocab=restrict_vocab
        )

    def most_similar_cosmul(self, positive=None, negative=None, topn=10):
        positive = [] if positive is None else positive
        negative = [] if negative is None else negative
        if isinstance(positive, (str, np.ndarray)):
            positive = [positive]
        if isinstance(negative, (str, np.ndarray)):
            negative = [negative]
        normed = self.get_normed_vectors()
        excluded = set()
        numerator = np.ones(len(self), dtype=np.float32)
        denominator = np.ones(len(self), dtype=np.float32)
        for items, accumulator in ((positive, numerator), (negative, denominator)):
            for item in items:
                weight = 1.0
                if isinstance(item, tuple):
                    item, weight = item
                if isinstance(item, np.ndarray):
                    vector = f32(item)
                    vector /= max(np.linalg.norm(vector), 1e-30)
                else:
                    excluded.add(self.get_index(item))
                    vector = self.get_vector(item, norm=True)
                similarity = (1.0 + dot_rows(normed, vector)) / 2.0
                accumulator *= similarity ** float(weight)
        scores = numerator / (denominator + 1e-6)
        result = []
        for index in np.argsort(-scores, kind="stable"):
            if int(index) not in excluded:
                result.append((self.index_to_key[int(index)], float(scores[index])))
                if len(result) >= int(topn):
                    break
        return result

    def similarity(self, w1, w2):
        a = self.get_vector(w1)
        b = self.get_vector(w2)
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denominator) if denominator else 0.0

    def distance(self, w1, w2):
        return 1.0 - self.similarity(w1, w2)

    def n_similarity(self, ws1, ws2):
        if not ws1 or not ws2:
            raise ZeroDivisionError("At least one of the passed list is empty.")
        a = np.mean(self[ws1], axis=0)
        b = np.mean(self[ws2], axis=0)
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denominator) if denominator else 0.0

    def distances(self, word_or_vector, other_words=()):
        vector = (
            self.get_vector(word_or_vector)
            if not isinstance(word_or_vector, np.ndarray)
            else word_or_vector
        )
        keys = self.index_to_key if len(other_words) == 0 else list(other_words)
        return np.asarray([1.0 - self.similarity_vectors(vector, self[key]) for key in keys])

    def doesnt_match(self, words):
        present = [word for word in words if word in self]
        if not present:
            raise ValueError("cannot select a mismatch from no in-vocabulary words")
        mean = self.get_mean_vector(
            present, pre_normalize=True, post_normalize=True
        )
        scores = [self.similarity_vectors(self.get_vector(word, norm=True), mean) for word in present]
        return present[int(np.argmin(scores))]

    def closer_than(self, key1, key2):
        threshold = self.distance(key1, key2)
        return [
            key
            for key, distance in zip(self.index_to_key, self.distances(key1))
            if key != key1 and distance < threshold
        ]

    def rank(self, key1, key2):
        ranked = self.most_similar(key1, topn=len(self))
        for position, (key, _) in enumerate(ranked, start=1):
            if key == key2:
                return position
        return len(self)

    def relative_cosine_similarity(self, wa, wb, topn=10):
        similarity = self.similarity(wa, wb)
        neighbors = self.most_similar(wa, topn=topn)
        scale = sum(score for _, score in neighbors)
        return similarity / scale if scale else 0.0

    @staticmethod
    def similarity_vectors(a, b):
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denominator) if denominator else 0.0

    def set_vecattr(self, key, attr, value):
        index = self.get_index(key)
        if attr not in self.expandos:
            self.expandos[attr] = np.zeros(len(self))
        self.expandos[attr][index] = value

    def get_vecattr(self, key, attr):
        return self.expandos[attr][self.get_index(key)].item()

    def vectors_for_all(self, keys, allow_inference=True, copy_vecattrs=False):
        result = KeyedVectors(self.vector_size)
        selected = []
        seen = set()
        for key in keys:
            if key in self and key not in seen:
                selected.append(key)
                seen.add(key)
        if selected:
            result.add_vectors(selected, self[selected])
        if copy_vecattrs:
            for attr in self.expandos:
                for key in selected:
                    result.set_vecattr(key, attr, self.get_vecattr(key, attr))
        return result

    def save_word2vec_format(
        self,
        fname,
        fvocab=None,
        binary=False,
        total_vec=None,
        write_header=True,
        prefix="",
        append=False,
        sort_attr="count",
    ):
        total = len(self) if total_vec is None else min(int(total_vec), len(self))
        mode = "ab" if append else "wb"
        with open(fname, mode) as stream:
            if write_header:
                stream.write(f"{total} {self.vector_size}\n".encode("utf-8"))
            for key, vector in zip(self.index_to_key[:total], self.vectors[:total]):
                encoded = f"{prefix}{key}".encode("utf-8")
                if binary:
                    stream.write(encoded + b" " + np.asarray(vector, dtype=np.float32).tobytes() + b"\n")
                else:
                    values = " ".join(f"{float(value):.9g}" for value in vector)
                    stream.write(encoded + b" " + values.encode("ascii") + b"\n")
        if fvocab is not None:
            with open(fvocab, "w", encoding="utf-8") as stream:
                for key in self.index_to_key[:total]:
                    count = self.get_vecattr(key, "count") if "count" in self.expandos else 1
                    stream.write(f"{prefix}{key} {count}\n")

    @classmethod
    def load_word2vec_format(
        cls,
        fname,
        fvocab=None,
        binary=False,
        encoding="utf8",
        unicode_errors="strict",
        limit=None,
        datatype=np.float32,
        no_header=False,
        binary_chunk_size=100 * 1024,
    ):
        with open(fname, "rb") as stream:
            if no_header:
                lines = stream.readlines()
                first = lines[0].decode(encoding, unicode_errors).split()
                vector_size = len(first) - 1
                rows = lines
                count = len(rows)
            else:
                count, vector_size = map(int, stream.readline().split())
                rows = None
            if limit is not None:
                count = min(count, int(limit))
            result = cls(vector_size)
            keys = []
            vectors = np.empty((count, vector_size), dtype=datatype)
            if binary:
                for row in range(count):
                    word = bytearray()
                    while True:
                        char = stream.read(1)
                        if not char:
                            raise EOFError("unexpected end of binary word2vec file")
                        if char == b" ":
                            break
                        if char != b"\n":
                            word.extend(char)
                    raw = stream.read(4 * vector_size)
                    if len(raw) != 4 * vector_size:
                        raise EOFError("truncated binary word2vec vector")
                    vectors[row] = np.frombuffer(raw, dtype=np.float32)
                    keys.append(word.decode(encoding, unicode_errors))
            else:
                source = rows if rows is not None else stream
                for row, line in enumerate(source):
                    if row >= count:
                        break
                    fields = line.decode(encoding, unicode_errors).rstrip().split()
                    keys.append(fields[0])
                    vectors[row] = np.asarray(fields[1:], dtype=datatype)
            result.add_vectors(keys, vectors[: len(keys)])
            return result

    def intersect_word2vec_format(
        self,
        fname,
        lockf=0.0,
        binary=False,
        encoding="utf8",
        unicode_errors="strict",
    ):
        source = self.load_word2vec_format(
            fname,
            binary=binary,
            encoding=encoding,
            unicode_errors=unicode_errors,
        )
        for key in source.index_to_key:
            if key in self:
                self.vectors[self.get_index(key)] = source[key]
        self.norms = None

    def save(self, fname_or_handle, **kwargs):
        if hasattr(fname_or_handle, "write"):
            pickle.dump(self, fname_or_handle)
        else:
            with open(fname_or_handle, "wb") as stream:
                pickle.dump(self, stream)

    @classmethod
    def load(cls, fname_or_handle, **kwargs):
        if hasattr(fname_or_handle, "read"):
            return pickle.load(fname_or_handle)
        with open(fname_or_handle, "rb") as stream:
            return pickle.load(stream)


def pseudorandom_weak_vector(size, seed_string=None, hashfxn=hash):
    seed = hashfxn(seed_string) & 0xFFFFFFFF if seed_string is not None else 0
    return np.random.default_rng(seed).uniform(-0.5 / size, 0.5 / size, size).astype(
        np.float32
    )
