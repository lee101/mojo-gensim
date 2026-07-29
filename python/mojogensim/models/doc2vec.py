"""Paragraph Vector (PV-DBOW and PV-DM) over the Mojo training kernels."""

from __future__ import annotations

from collections import namedtuple

import numpy as np

from .._lib import addr, i64, lib
from .keyedvectors import KeyedVectors
from .word2vec import Word2Vec

TaggedDocument = namedtuple("TaggedDocument", "words tags")
LabeledSentence = TaggedDocument


class Doc2Vec(Word2Vec):
    def __init__(
        self,
        documents=None,
        corpus_file=None,
        vector_size=100,
        dm_mean=None,
        dm=1,
        dbow_words=0,
        dm_concat=0,
        dm_tag_count=1,
        dv=None,
        dv_mapfile=None,
        comment=None,
        trim_rule=None,
        callbacks=(),
        window=5,
        epochs=10,
        **kwargs,
    ):
        if dm_concat:
            raise NotImplementedError("dm_concat is not covered")
        super().__init__(
            sentences=None,
            corpus_file=corpus_file,
            vector_size=vector_size,
            window=window,
            epochs=epochs,
            callbacks=callbacks,
            sg=0,
            **kwargs,
        )
        self.dm = int(dm)
        self.dbow_words = int(dbow_words)
        self.dm_concat = int(dm_concat)
        self.dm_tag_count = int(dm_tag_count)
        self.cbow_mean = int(dm_mean) if dm_mean is not None else 1
        self.dv = dv if dv is not None else KeyedVectors(self.vector_size)
        self.docvecs = self.dv
        if documents is not None:
            cached = [self._as_tagged(doc) for doc in documents]
            self.build_vocab(cached)
            self.train(cached, total_examples=self.corpus_count, epochs=self.epochs)

    @staticmethod
    def _as_tagged(doc):
        return doc if hasattr(doc, "words") and hasattr(doc, "tags") else TaggedDocument(*doc)

    def build_vocab(self, corpus_iterable=None, corpus_file=None, **kwargs):
        if corpus_iterable is None:
            raise NotImplementedError("pass corpus_iterable")
        docs = [self._as_tagged(doc) for doc in corpus_iterable]
        super().build_vocab([doc.words for doc in docs], corpus_file, **kwargs)
        tags = []
        seen = set()
        for doc in docs:
            for tag in doc.tags:
                if tag not in seen:
                    tags.append(tag)
                    seen.add(tag)
        self.dv = KeyedVectors(self.vector_size)
        if tags:
            self.dv.add_vectors(tags, self._initial_doc_vectors(len(tags)))
        self.dv.vectors_lockf = np.ones(1, dtype=np.float32)
        self.docvecs = self.dv

    def _initial_doc_vectors(self, count):
        return np.random.default_rng(self.seed + 7919).uniform(
            -0.5 / self.vector_size,
            0.5 / self.vector_size,
            size=(count, self.vector_size),
        ).astype(np.float32)

    def _encoded_docs(self, docs):
        encoded = []
        for doc in docs:
            words = [
                self.wv.key_to_index[word] for word in doc.words if word in self.wv
            ]
            tags = [self.dv.get_index(tag) for tag in doc.tags if tag in self.dv]
            for tag in tags:
                encoded.append((words, tag))
        return encoded

    def _dbow_epoch(self, encoded, rng, alpha, doc_vectors=None, outputs=None):
        docs = []
        targets = []
        for words, doc_id in encoded:
            docs.extend([doc_id] * len(words))
            targets.extend(words)
        if not targets:
            return 0
        doc_array = i64(docs)
        target_array = i64(targets)
        negatives = self._negative_samples(targets, rng)
        lib().mg_sgns_train(
            addr(self.dv.vectors if doc_vectors is None else doc_vectors, np.float32, writable=True),
            addr(self.syn1neg if outputs is None else outputs, np.float32, writable=True),
            addr(doc_array, np.int64),
            addr(target_array, np.int64),
            addr(negatives, np.int64),
            len(targets),
            self.vector_size,
            self.negative,
            alpha,
        )
        return len(targets)

    def _dm_epoch(
        self,
        encoded,
        rng,
        alpha,
        doc_vectors=None,
        words=None,
        outputs=None,
        update_words=True,
    ):
        contexts = []
        indptr = [0]
        docs = []
        targets = []
        for sentence, doc_id in encoded:
            for pos, target in enumerate(sentence):
                left = max(0, pos - self.window)
                right = min(len(sentence), pos + self.window + 1)
                context = [
                    sentence[index] for index in range(left, right) if index != pos
                ]
                contexts.extend(context)
                indptr.append(len(contexts))
                docs.append(doc_id)
                targets.append(target)
        if not targets:
            return 0
        context_array = i64(contexts if contexts else [0])
        indptr_array = i64(indptr)
        doc_array = i64(docs)
        target_array = i64(targets)
        negatives = self._negative_samples(targets, rng)
        hidden = np.empty(self.vector_size, dtype=np.float32)
        gradient = np.empty(self.vector_size, dtype=np.float32)
        lib().mg_dm_train(
            addr(self.wv.vectors if words is None else words, np.float32, writable=update_words),
            addr(self.dv.vectors if doc_vectors is None else doc_vectors, np.float32, writable=True),
            addr(self.syn1neg if outputs is None else outputs, np.float32, writable=True),
            addr(context_array, np.int64),
            addr(indptr_array, np.int64),
            addr(doc_array, np.int64),
            addr(target_array, np.int64),
            addr(negatives, np.int64),
            addr(hidden, np.float32, writable=True),
            addr(gradient, np.float32, writable=True),
            len(targets),
            self.vector_size,
            self.negative,
            alpha,
            self.cbow_mean,
            int(update_words),
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
        **kwargs,
    ):
        if corpus_iterable is None:
            raise NotImplementedError("pass corpus_iterable")
        docs = [self._as_tagged(doc) for doc in corpus_iterable]
        encoded = self._encoded_docs(docs)
        epochs = self.epochs if epochs is None else int(epochs)
        start = self.alpha if start_alpha is None else float(start_alpha)
        end = self.min_alpha if end_alpha is None else float(end_alpha)
        rng = np.random.default_rng(self.seed)
        trained = 0
        for epoch in range(epochs):
            fraction = epoch / max(1, epochs - 1)
            alpha = start + fraction * (end - start)
            if self.dm:
                trained += self._dm_epoch(encoded, rng, alpha)
            else:
                trained += self._dbow_epoch(encoded, rng, alpha)
                if self.dbow_words:
                    trained += self._sg_batch([words for words, _ in encoded], rng, alpha)
        self.dv.norms = None
        self.wv.norms = None
        return trained, sum(len(doc.words) for doc in docs) * epochs

    def infer_vector(
        self,
        doc_words,
        alpha=None,
        min_alpha=None,
        epochs=None,
    ):
        words = [self.wv.key_to_index[word] for word in doc_words if word in self.wv]
        if not words:
            return np.zeros(self.vector_size, dtype=np.float32)
        inferred = self._initial_doc_vectors(1)
        output_copy = self.syn1neg.copy()
        word_copy = self.wv.vectors.copy()
        epochs = self.epochs if epochs is None else int(epochs)
        start = self.alpha if alpha is None else float(alpha)
        end = self.min_alpha if min_alpha is None else float(min_alpha)
        rng = np.random.default_rng(self.seed + 104729)
        encoded = [(words, 0)]
        for epoch in range(epochs):
            fraction = epoch / max(1, epochs - 1)
            rate = start + fraction * (end - start)
            if self.dm:
                self._dm_epoch(
                    encoded,
                    rng,
                    rate,
                    doc_vectors=inferred,
                    words=word_copy,
                    outputs=output_copy,
                    update_words=False,
                )
            else:
                self._dbow_epoch(
                    encoded,
                    rng,
                    rate,
                    doc_vectors=inferred,
                    outputs=output_copy,
                )
        return inferred[0]
