"""Word2Vec with Mojo skip-gram and CBOW negative-sampling kernels."""

from __future__ import annotations

from collections import Counter
import pickle

import numpy as np

from .._lib import addr, f32, i64, lib
from .keyedvectors import KeyedVectors


class Word2Vec:
    def __init__(
        self,
        sentences=None,
        corpus_file=None,
        vector_size=100,
        alpha=0.025,
        window=5,
        min_count=5,
        max_vocab_size=None,
        sample=1e-3,
        seed=1,
        workers=3,
        min_alpha=0.0001,
        sg=0,
        hs=0,
        negative=5,
        ns_exponent=0.75,
        cbow_mean=1,
        hashfxn=hash,
        epochs=5,
        null_word=0,
        trim_rule=None,
        sorted_vocab=1,
        batch_words=10000,
        compute_loss=False,
        callbacks=(),
        comment=None,
        max_final_vocab=None,
        **kwargs,
    ):
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported Word2Vec arguments: {names}")
        if corpus_file is not None:
            raise NotImplementedError("corpus_file training is not covered; pass sentences")
        if hs:
            raise NotImplementedError("hierarchical softmax is not covered")
        if compute_loss:
            raise NotImplementedError("training-loss computation is not covered")
        self.vector_size = int(vector_size)
        if self.vector_size <= 0:
            raise ValueError("vector_size must be positive")
        self.layer1_size = self.vector_size
        self.alpha = float(alpha)
        self.min_alpha = float(min_alpha)
        self.window = int(window)
        self.min_count = int(min_count)
        self.max_vocab_size = max_vocab_size
        self.sample = float(sample)
        self.seed = int(seed)
        self.workers = int(workers)
        self.sg = int(sg)
        self.hs = int(hs)
        self.negative = int(negative)
        self.ns_exponent = float(ns_exponent)
        self.cbow_mean = int(cbow_mean)
        self.epochs = int(epochs)
        if not np.isfinite(self.alpha) or not np.isfinite(self.min_alpha):
            raise ValueError("alpha and min_alpha must be finite")
        if self.window < 0 or self.min_count < 0 or self.negative < 0:
            raise ValueError("window, min_count, and negative must be non-negative")
        if self.epochs < 0:
            raise ValueError("epochs must be non-negative")
        if self.sg not in (0, 1) or self.cbow_mean not in (0, 1):
            raise ValueError("sg and cbow_mean must be either 0 or 1")
        self.sorted_vocab = int(sorted_vocab)
        self.batch_words = int(batch_words)
        self.compute_loss = bool(compute_loss)
        self.callbacks = tuple(callbacks)
        self.wv = KeyedVectors(self.vector_size)
        self.syn1neg = np.empty((0, self.vector_size), dtype=np.float32)
        self.raw_vocab: dict = {}
        self.corpus_count = 0
        self.corpus_total_words = 0
        self._rng = np.random.default_rng(self.seed)
        self._negative_probs = np.empty(0)
        self._negative_alias_prob = np.empty(0)
        self._negative_alias = np.empty(0, dtype=np.int64)
        self._keep_probs = np.empty(0)
        self._latest_training_loss = 0.0
        if sentences is not None:
            cached = [list(sentence) for sentence in sentences]
            self.build_vocab(cached)
            self.train(
                cached,
                total_examples=self.corpus_count,
                epochs=self.epochs,
            )

    def scan_vocab(self, corpus_iterable, progress_per=10000, trim_rule=None):
        counts = Counter()
        sentence_count = 0
        total_words = 0
        for sentence in corpus_iterable:
            counts.update(sentence)
            sentence_count += 1
            total_words += len(sentence)
        self.raw_vocab = dict(counts)
        return total_words, sentence_count

    def build_vocab(
        self,
        corpus_iterable=None,
        corpus_file=None,
        update=False,
        progress_per=10000,
        keep_raw_vocab=False,
        trim_rule=None,
        **kwargs,
    ):
        if corpus_file is not None or corpus_iterable is None:
            raise NotImplementedError("pass corpus_iterable")
        docs = [list(sentence) for sentence in corpus_iterable]
        total, count = self.scan_vocab(docs, progress_per, trim_rule)
        retained = [
            (word, freq, order)
            for order, (word, freq) in enumerate(self.raw_vocab.items())
            if freq >= self.min_count
        ]
        if self.sorted_vocab:
            retained.sort(key=lambda item: (item[1], item[2]), reverse=True)
        if self.max_vocab_size is not None:
            retained = retained[: self.max_vocab_size]
        retained = [(word, freq) for word, freq, _ in retained]
        if update:
            old = set(self.wv.key_to_index)
            new_items = [(word, freq) for word, freq in retained if word not in old]
            if new_items:
                vectors = self._initial_vectors([word for word, _ in new_items])
                self.wv.add_vectors([word for word, _ in new_items], vectors)
                self.syn1neg = np.vstack(
                    (
                        self.syn1neg,
                        np.zeros((len(new_items), self.vector_size), dtype=np.float32),
                    )
                )
            for word, freq in retained:
                self.wv.set_vecattr(
                    word,
                    "count",
                    freq + (self.wv.get_vecattr(word, "count") if word in old else 0),
                )
        else:
            self.wv = KeyedVectors(self.vector_size)
            words = [word for word, _ in retained]
            if words:
                self.wv.add_vectors(words, self._initial_vectors(words))
            self.wv.expandos["count"] = np.asarray(
                [freq for _, freq in retained], dtype=np.int64
            )
            self.syn1neg = np.zeros(
                (len(retained), self.vector_size), dtype=np.float32
            )
        self.wv.vectors_lockf = np.ones(1, dtype=np.float32)
        counts = self.wv.expandos.get("count", np.empty(0))
        if len(counts):
            weights = np.asarray(counts, dtype=np.float64) ** self.ns_exponent
            self._negative_probs = weights / weights.sum()
            self._build_negative_alias()
            if self.sample > 0 and total:
                threshold = self.sample * total
                self._keep_probs = np.minimum(
                    1.0,
                    (np.sqrt(np.asarray(counts) / threshold) + 1.0)
                    * threshold
                    / np.asarray(counts),
                )
            else:
                self._keep_probs = np.ones(len(counts))
        self.corpus_count = count
        self.corpus_total_words = total
        if not keep_raw_vocab:
            self.raw_vocab = {}
        return None

    def _initial_vectors(self, words):
        rng = np.random.default_rng(self.seed)
        return rng.uniform(
            -0.5 / self.vector_size,
            0.5 / self.vector_size,
            size=(len(words), self.vector_size),
        ).astype(np.float32)

    def _encode(self, corpus, rng=None):
        encoded = []
        indices = self.wv.key_to_index
        for sentence in corpus:
            row = []
            for word in sentence:
                index = indices.get(word)
                if index is None:
                    continue
                if rng is None or rng.random() < self._keep_probs[index]:
                    row.append(index)
            encoded.append(row)
        return encoded

    def _build_negative_alias(self):
        count = len(self._negative_probs)
        scaled = self._negative_probs * count
        probabilities = np.empty(count, dtype=np.float64)
        aliases = np.empty(count, dtype=np.int64)
        small = list(np.flatnonzero(scaled < 1.0))
        large = list(np.flatnonzero(scaled >= 1.0))
        while small and large:
            low = small.pop()
            high = large.pop()
            probabilities[low] = scaled[low]
            aliases[low] = high
            scaled[high] -= 1.0 - probabilities[low]
            (small if scaled[high] < 1.0 else large).append(high)
        for index in small + large:
            probabilities[index] = 1.0
            aliases[index] = index
        self._negative_alias_prob = probabilities
        self._negative_alias = aliases

    def _draw_negative(self, rng, size):
        result = rng.integers(
            0,
            len(self._negative_alias),
            size=size,
            dtype=np.int64,
        )
        rejected = rng.random(size) >= self._negative_alias_prob[result]
        result[rejected] = self._negative_alias[result[rejected]]
        return result

    def _negative_samples(self, targets, rng):
        count = len(targets)
        width = max(1, self.negative)
        result = self._draw_negative(rng, (count, width))
        if self.negative and len(self.wv) > 1:
            target_array = np.asarray(targets)
            clashes = result[:, : self.negative] == target_array[:, None]
            while np.any(clashes):
                result[:, : self.negative][clashes] = self._draw_negative(
                    rng, int(clashes.sum())
                )
                clashes = result[:, : self.negative] == target_array[:, None]
        return np.ascontiguousarray(result)

    def _sg_batch(self, encoded, rng, alpha):
        if self.window <= 0:
            return 0
        lengths = np.fromiter(
            (len(sentence) for sentence in encoded),
            dtype=np.int64,
            count=len(encoded),
        )
        indptr = np.empty(len(encoded) + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(lengths, out=indptr[1:])
        token_count = int(indptr[-1])
        if token_count == 0:
            return 0
        tokens = np.fromiter(
            (token for sentence in encoded for token in sentence),
            dtype=np.int64,
            count=token_count,
        )
        reduced = rng.integers(
            0,
            self.window,
            size=token_count,
            dtype=np.int64,
        )
        capacity = token_count * 2 * self.window
        center_array = np.empty(capacity, dtype=np.int64)
        target_array = np.empty(capacity, dtype=np.int64)
        pair_count = lib().mg_sg_pairs(
            addr(tokens, np.int64),
            addr(indptr, np.int64),
            addr(reduced, np.int64),
            addr(center_array, np.int64, writable=True),
            addr(target_array, np.int64, writable=True),
            len(encoded),
            self.window,
        )
        if pair_count == 0:
            return 0
        centers = center_array[:pair_count]
        targets = target_array[:pair_count]
        negatives = self._negative_samples(targets, rng)
        lib().mg_sgns_train(
            addr(self.wv.vectors, np.float32, writable=True),
            addr(self.syn1neg, np.float32, writable=True),
            addr(centers, np.int64),
            addr(targets, np.int64),
            addr(negatives, np.int64),
            pair_count,
            self.vector_size,
            self.negative,
            alpha,
        )
        return pair_count

    def _cbow_batch(self, encoded, rng, alpha):
        contexts = []
        indptr = [0]
        targets = []
        for sentence in encoded:
            for pos, target in enumerate(sentence):
                reduced = (
                    int(rng.integers(0, self.window))
                    if self.window > 0
                    else 0
                )
                left = max(0, pos - self.window + reduced)
                right = min(len(sentence), pos + self.window + 1 - reduced)
                context = [
                    sentence[index]
                    for index in range(left, right)
                    if index != pos
                ]
                if context:
                    contexts.extend(context)
                    indptr.append(len(contexts))
                    targets.append(target)
        if not targets:
            return 0
        context_array = i64(contexts)
        indptr_array = i64(indptr)
        target_array = i64(targets)
        negatives = self._negative_samples(targets, rng)
        hidden = np.empty(self.vector_size, dtype=np.float32)
        gradient = np.empty(self.vector_size, dtype=np.float32)
        lib().mg_cbow_train(
            addr(self.wv.vectors, np.float32, writable=True),
            addr(self.syn1neg, np.float32, writable=True),
            addr(context_array, np.int64),
            addr(indptr_array, np.int64),
            addr(target_array, np.int64),
            addr(negatives, np.int64),
            addr(hidden, np.float32, writable=True),
            addr(gradient, np.float32, writable=True),
            len(targets),
            self.vector_size,
            self.negative,
            alpha,
            self.cbow_mean,
        )
        return len(targets)

    def train(
        self,
        corpus_iterable=None,
        corpus_file=None,
        total_examples=None,
        total_words=None,
        epochs=None,
        start_alpha=None,
        end_alpha=None,
        word_count=0,
        queue_factor=2,
        report_delay=1.0,
        compute_loss=False,
        callbacks=(),
        **kwargs,
    ):
        if corpus_file is not None or corpus_iterable is None:
            raise NotImplementedError("pass corpus_iterable")
        if not len(self.wv):
            return 0, 0
        docs = [list(sentence) for sentence in corpus_iterable]
        epochs = self.epochs if epochs is None else int(epochs)
        start = self.alpha if start_alpha is None else float(start_alpha)
        end = self.min_alpha if end_alpha is None else float(end_alpha)
        if epochs < 0 or not np.isfinite(start) or not np.isfinite(end):
            raise ValueError("epochs must be non-negative and learning rates finite")
        rng = np.random.default_rng(self.seed)
        trained = 0
        active_callbacks = tuple(self.callbacks) + tuple(callbacks)
        for callback in active_callbacks:
            method = getattr(callback, "on_train_begin", None)
            if method:
                method(self)
        for epoch in range(epochs):
            for callback in active_callbacks:
                method = getattr(callback, "on_epoch_begin", None)
                if method:
                    method(self)
            fraction = epoch / max(1, epochs - 1)
            learning_rate = start + fraction * (end - start)
            encoded = self._encode(docs, rng)
            trained += (
                self._sg_batch(encoded, rng, learning_rate)
                if self.sg
                else self._cbow_batch(encoded, rng, learning_rate)
            )
            for callback in active_callbacks:
                method = getattr(callback, "on_epoch_end", None)
                if method:
                    method(self)
        for callback in active_callbacks:
            method = getattr(callback, "on_train_end", None)
            if method:
                method(self)
        self.wv.norms = None
        return trained, sum(map(len, docs)) * epochs

    def get_latest_training_loss(self):
        return self._latest_training_loss

    def predict_output_word(self, context_words_list, topn=10):
        indices = [self.wv.get_index(word) for word in context_words_list]
        hidden = self.wv.vectors[indices].mean(axis=0)
        logits = self.syn1neg @ hidden
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        order = np.argsort(-probabilities)[:topn]
        return [(self.wv.index_to_key[i], float(probabilities[i])) for i in order]

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
