# mojo-gensim

`mojo-gensim` is a standalone Mojo port of the compute-heavy core of
[gensim](https://radimrehurek.com/gensim/): training dense word and document
embeddings, projecting sparse corpora with LSI, and exact cosine-similarity
queries. The Python package is named `mojogensim`, so it can be installed next
to upstream gensim for parity testing, but the covered classes keep gensim's
names and the documented subset of common constructor arguments, attributes,
and query methods. Unknown constructor arguments fail instead of being ignored.

This is an implementation, not a binding to gensim. Training and sparse matrix
products execute in Mojo through a small C ABI; NumPy owns the arrays and the
Python layer handles vocabulary and corpus bookkeeping.

## Covered subset

| upstream area | implemented |
| --- | --- |
| `gensim.models.KeyedVectors` | keyed storage and attributes, normalized vectors, cosine/distance queries, weighted `most_similar`, `most_similar_cosmul`, mismatch selection, pickle, and word2vec text/binary round trips |
| `gensim.models.Word2Vec` | vocabulary order/counts, skip-gram or CBOW training with negative sampling, dynamic windows, deterministic training, and `predict_output_word` |
| `gensim.models.Doc2Vec` | `TaggedDocument`, PV-DBOW and PV-DM training, document vectors, and deterministic `infer_vector` |
| `gensim.models.LsiModel` | randomized truncated SVD of sparse bag-of-words corpora, power iterations, corpus/document projection, incremental refit, topics, save/load |
| `gensim.similarities` | exact `MatrixSimilarity`, `SparseMatrixSimilarity`-shaped and `Similarity`-shaped indexes, dense or bag-of-words queries, `num_best`, adding documents |
| `gensim.matutils` | `sparse2full`, `full2sparse`, `unitvec`, `corpus2dense`, `dense2vec`, and `argsort` |

The numerical tests use the installed gensim itself, not hand-picked expected values.
They compare vocabulary order and counts, cosine scores and ranking, LSI
singular values/subspaces and projections, similarity-index results, corpus
utilities, and public serialization. A separate scalar reference checks every
update performed by the Mojo SGNS kernel.

## Not covered

Hierarchical softmax, FastText, phrases, topic models other than LSI,
Word Mover's Distance, distributed training, and gensim's downloader are
outside this repository. Word2Vec and Doc2Vec accept `workers` for source
compatibility but currently train in one deterministic process; `corpus_file`
and `dm_concat` are not implemented. Training therefore optimizes the same
negative-sampling objectives as gensim but is not expected to produce
bit-identical embeddings from the same seed.

LSI is an in-memory randomized SVD, not gensim's streaming stochastic update.
`add_documents` retains and refits the accumulated corpus, so it is appropriate
for medium corpora rather than unbounded streams. The similarity classes use
an exact dense float32 index; the `SparseMatrixSimilarity` name is accepted,
but it does not preserve sparse storage or shard to disk.

## Install

The repository pins its own Mojo nightly and all Python dependencies:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-gensim.so`. The Python package gives a
clear error if that library has not been built.

## Usage

This example is also suitable to paste into `pixi run python`:

```python
from mojogensim.models import Word2Vec

sentences = [
    ["king", "queen", "royal"],
    ["queen", "crown", "royal"],
    ["dog", "cat", "pet"],
] * 100

model = Word2Vec(
    sentences,
    vector_size=32,
    window=2,
    min_count=1,
    sg=1,
    negative=5,
    sample=0,
    epochs=10,
    workers=1,
    seed=7,
)

print(model.wv.most_similar("king", topn=3))
print(model.wv.similarity("king", "queen"))
```

The other covered entry points use their upstream names:

```python
from mojogensim.models import Doc2Vec, LsiModel, TaggedDocument
from mojogensim.similarities import MatrixSimilarity

docs = [
    TaggedDocument(["red", "green", "color"], ["colors"]),
    TaggedDocument(["dog", "cat", "pet"], ["animals"]),
]
paragraphs = Doc2Vec(docs, vector_size=24, min_count=1, epochs=20)
inferred = paragraphs.infer_vector(["red", "color"])

corpus = [[(0, 1.0), (1, 2.0)], [(0, 2.0), (2, 1.0)]]
lsi = LsiModel(corpus, id2word={0: "a", 1: "b", 2: "c"}, num_topics=2)
index = MatrixSimilarity(corpus, num_features=3)
scores = index[[(0, 1.0)]]
```

## Benchmarks

Measured by `pixi run bench`, which holds a machine-wide lock. These are
best-of-three public-API timings on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64, Python 3.13.14, NumPy 2.5.1, and gensim 4.4.0:

| case | mojo-gensim | gensim | result |
| --- | ---: | ---: | ---: |
| KeyedVectors.most_similar (50k x 128) | 4.9 ms | 2.8 ms | 1.76x slower |
| MatrixSimilarity query (40k x 96) | 2.7 ms | 3.0 ms | 1.09x faster |
| Word2Vec SGNS fit (4k x 16, 2 epochs) | 380.9 ms | 635.3 ms | 1.67x faster |
| LSI fit (4k docs, 2k terms, 40 topics) | 670.1 ms | 690.5 ms | 1.03x faster |

LSI does repeated CSR-by-dense products in compiled Mojo and reduces the final
dense SVD to the randomized sketch. Cosine queries fuse row scaling into
four-accumulator SIMD scoring, cache row norms, use thresholded four-worker
chunks, and select top results without fully sorting every score. SGNS builds
dynamic-window pairs directly in NumPy-owned buffers, samples negatives through
an alias table, and applies vector updates with SIMD.

No GPU path is included. The hot kernels are memory-bound: cosine scoring is
about 0.5 flop/byte, and SGNS and CSR products remain well below the 2 flop/byte
GPU threshold after update and result traffic are counted.

Run the same benchmark on another machine with:

```bash
pixi run bench
```

## How it works

All kernels live in one compilation unit, `src/kernels.mojo`, because this
nightly's shared-library build cost is mostly fixed. Exports use
`@export("name")` and `abi("C")`; ctypes passes every NumPy buffer as an integer
address, and Mojo reconstructs it as an
`UnsafePointer[..., AnyOrigin[mut=True]]`.

Embedding matrices and similarity indexes are C-contiguous, row-major
float32 arrays. Cosine normalization and row scoring use SIMD and parallelize
large row sets in contiguous chunks. Keyed-vector queries score unnormalized
rows against cached norms, avoiding a full normalized-matrix allocation. SGNS
dynamic-window pairs are written directly into NumPy buffers by Mojo, while
CBOW, PV-DBOW, and PV-DM receive precomputed integer context/negative-sample
arrays and update the input/output matrices in place.

LSI corpora cross as float64 CSR: `indptr`, `indices`, and `values`. Mojo
computes both `X @ dense` and `X.T @ dense`; NumPy performs only the small QR
factorizations and SVD of the randomized sketch. Mojo never retains or
allocates Python-owned memory, so array lifetime remains entirely on the Python
side.

## License

MIT
