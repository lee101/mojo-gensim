"""Public-API benchmarks against gensim on identical inputs."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import gensim  # noqa: E402
from gensim.models import KeyedVectors as GKeyedVectors  # noqa: E402
from gensim.models import LsiModel as GLsiModel  # noqa: E402
from gensim.models import Word2Vec as GWord2Vec  # noqa: E402
from gensim.similarities import MatrixSimilarity as GMatrixSimilarity  # noqa: E402

from mojogensim.models import KeyedVectors, LsiModel, Word2Vec  # noqa: E402
from mojogensim.similarities import MatrixSimilarity  # noqa: E402


def timeit(function, repeat=3):
    function()
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def format_time(seconds):
    if seconds < 0.001:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def keyed_vectors_case():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(50_000, 128)).astype(np.float32)
    keys = [f"word-{index}" for index in range(len(values))]
    ours = KeyedVectors(128)
    theirs = GKeyedVectors(128)
    ours.add_vectors(keys, values)
    theirs.add_vectors(keys, values)
    return (
        lambda: ours.most_similar("word-123", topn=20),
        lambda: theirs.most_similar("word-123", topn=20),
    )


def matrix_similarity_case():
    rng = np.random.default_rng(2)
    corpus = rng.normal(size=(40_000, 96)).astype(np.float32)
    query = rng.normal(size=96).astype(np.float32)
    ours = MatrixSimilarity(corpus, num_features=96, num_best=20)
    theirs = GMatrixSimilarity(corpus, num_features=96, num_best=20)
    return lambda: ours[query], lambda: theirs[query]


def word2vec_case():
    rng = np.random.default_rng(3)
    vocab = [f"w{index}" for index in range(2_000)]
    corpus = [
        [vocab[index] for index in rng.integers(0, len(vocab), size=16)]
        for _ in range(4_000)
    ]
    common = dict(
        vector_size=64,
        window=5,
        min_count=1,
        negative=5,
        sg=1,
        epochs=2,
        seed=4,
    )
    return (
        lambda: Word2Vec(corpus, workers=1, **common),
        lambda: GWord2Vec(corpus, workers=1, **common),
    )


def lsi_case():
    rng = np.random.default_rng(5)
    docs = 4_000
    terms = 2_000
    width = 30
    corpus = []
    for _ in range(docs):
        indices = np.sort(rng.choice(terms, size=width, replace=False))
        values = rng.random(width)
        corpus.append(list(zip(indices.tolist(), values.tolist())))
    id2word = {index: str(index) for index in range(terms)}
    common = dict(
        num_topics=40,
        id2word=id2word,
        power_iters=2,
        extra_samples=20,
        random_seed=6,
    )
    return (
        lambda: LsiModel(corpus, **common),
        lambda: GLsiModel(corpus, **common),
    )


CASES = [
    ("KeyedVectors.most_similar (50k x 128)", keyed_vectors_case),
    ("MatrixSimilarity query (40k x 96)", matrix_similarity_case),
    ("Word2Vec SGNS fit (4k x 16, 2 epochs)", word2vec_case),
    ("LSI fit (4k docs, 2k terms, 40 topics)", lsi_case),
]


def main():
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}")
    print(
        f"Python {platform.python_version()}, NumPy {np.__version__}, "
        f"gensim {gensim.__version__}"
    )
    print()
    print("| case | mojo-gensim | gensim | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, setup in CASES:
        ours, theirs = setup()
        ours_time = timeit(ours)
        theirs_time = timeit(theirs)
        ratio = theirs_time / ours_time
        result = (
            f"{ratio:.2f}x faster"
            if ratio >= 1
            else f"{1.0 / ratio:.2f}x slower"
        )
        print(
            f"| {name} | {format_time(ours_time)} | "
            f"{format_time(theirs_time)} | {result} |"
        )


if __name__ == "__main__":
    main()
