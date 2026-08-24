"""Numerical and behavioural parity with gensim 4.x on the covered subset."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

gensim = pytest.importorskip("gensim")
from gensim import matutils as gmat
from gensim.models import KeyedVectors as GKeyedVectors
from gensim.models import LsiModel as GLsiModel
from gensim.models import Word2Vec as GWord2Vec
from gensim.similarities import MatrixSimilarity as GMatrixSimilarity

from mojogensim import matutils
from mojogensim._lib import (
    addr,
    cosine_rows,
    dot_rows,
    f32,
    f64,
    i64,
    lib,
    normalize_rows,
)
from mojogensim.models import (
    Doc2Vec,
    KeyedVectors,
    LsiModel,
    TaggedDocument,
    Word2Vec,
)
from mojogensim.models.keyedvectors import _top_order
from mojogensim.similarities import (
    MatrixSimilarity,
    Similarity,
    SparseMatrixSimilarity,
)


@pytest.fixture
def vectors():
    rng = np.random.default_rng(4)
    keys = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    values = rng.normal(size=(len(keys), 13)).astype(np.float32)
    ours = KeyedVectors(13)
    theirs = GKeyedVectors(13)
    ours.add_vectors(keys, values)
    theirs.add_vectors(keys, values)
    return ours, theirs


def test_keyedvectors_storage_and_lookup(vectors):
    ours, theirs = vectors
    assert ours.index_to_key == theirs.index_to_key
    assert ours.key_to_index == theirs.key_to_index
    assert np.array_equal(ours["gamma"], theirs["gamma"])
    assert np.array_equal(ours[["alpha", "delta"]], theirs[["alpha", "delta"]])
    assert ours.get_index("missing", -1) == theirs.get_index("missing", -1)
    with pytest.raises(KeyError):
        ours.get_vector("missing")


@pytest.mark.filterwarnings("ignore:Adding single vectors:UserWarning")
def test_keyedvectors_preallocation_matches_upstream():
    ours = KeyedVectors(3, count=2)
    theirs = GKeyedVectors(3, count=2)
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert ours.add_vector("first", vector) == theirs.add_vector("first", vector)
    assert ours.index_to_key == theirs.index_to_key
    assert np.array_equal(ours.vectors, theirs.vectors)


def test_keyedvectors_cosine_parity(vectors):
    ours, theirs = vectors
    assert ours.similarity("alpha", "beta") == pytest.approx(
        theirs.similarity("alpha", "beta"), abs=2e-7
    )
    assert ours.distance("alpha", "beta") == pytest.approx(
        theirs.distance("alpha", "beta"), abs=2e-7
    )
    assert ours.n_similarity(["alpha", "beta"], ["delta", "zeta"]) == pytest.approx(
        theirs.n_similarity(["alpha", "beta"], ["delta", "zeta"]), abs=2e-7
    )


def test_most_similar_parity(vectors):
    ours, theirs = vectors
    a = ours.most_similar(
        positive=[("alpha", 0.7), "beta"], negative=["gamma"], topn=4
    )
    b = theirs.most_similar(
        positive=[("alpha", 0.7), "beta"], negative=["gamma"], topn=4
    )
    assert [key for key, _ in a] == [key for key, _ in b]
    assert np.allclose([score for _, score in a], [score for _, score in b], atol=2e-7)


@pytest.mark.filterwarnings("ignore:Call to deprecated `init_sims`:DeprecationWarning")
def test_keyedvectors_extended_queries(vectors):
    ours, theirs = vectors
    a = ours.most_similar_cosmul(
        positive=["alpha", "beta"], negative=["gamma"], topn=3
    )
    b = theirs.most_similar_cosmul(
        positive=["alpha", "beta"], negative=["gamma"], topn=3
    )
    assert [key for key, _ in a] == [key for key, _ in b]
    assert np.allclose([score for _, score in a], [score for _, score in b], atol=2e-6)
    assert ours.doesnt_match(["alpha", "beta", "delta"]) == theirs.doesnt_match(
        ["alpha", "beta", "delta"]
    )


@pytest.mark.parametrize("rows", [257, 31_497])
def test_mojo_row_normalization_matches_numpy(rows):
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=(rows, 127)).astype(np.float32)
    matrix[17] = 0
    actual = normalize_rows(matrix)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    expected = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)
    assert np.allclose(actual, expected, atol=2e-7)


def test_ffi_rejects_wrong_dtype_layout_mutability_and_narrowing():
    matrix = np.ones((3, 4), dtype=np.float32)
    with pytest.raises(TypeError, match="expected exactly float64"):
        addr(matrix, np.float64)
    with pytest.raises(TypeError, match="C-contiguous"):
        addr(matrix[:, ::2], np.float32)
    matrix.flags.writeable = False
    with pytest.raises(TypeError, match="writable"):
        addr(matrix, np.float32, writable=True)
    with pytest.raises(TypeError, match="complex"):
        f32(np.array([1 + 2j]))
    with pytest.raises(TypeError, match="complex"):
        f64(np.array([1 + 2j]))
    with pytest.raises(OverflowError, match="float32"):
        f32(np.array([np.finfo(np.float64).max]))
    with pytest.raises(OverflowError, match="int64"):
        i64(np.array([np.iinfo(np.uint64).max], dtype=np.uint64))


def test_invalid_kernel_parameters_fail_before_ffi():
    with pytest.raises(ValueError, match="vector_size"):
        Word2Vec(vector_size=0)
    with pytest.raises(ValueError, match="non-negative"):
        Word2Vec(vector_size=4, negative=-1)
    with pytest.raises(ValueError, match="num_topics"):
        LsiModel(num_topics=0)
    with pytest.raises(TypeError, match="float32"):
        MatrixSimilarity([[1.0, 0.0]], dtype=np.float64)
    with pytest.raises(TypeError, match="unsupported Word2Vec"):
        Word2Vec(vector_size=4, misspelled_option=True)
    with pytest.raises(NotImplementedError, match="loss"):
        Word2Vec(vector_size=4, compute_loss=True)


@pytest.mark.parametrize("rows", [7, 31_497])
def test_simd_tail_row_scoring_across_parallel_threshold(rows):
    rng = np.random.default_rng(15)
    matrix = rng.normal(size=(rows, 127)).astype(np.float32)
    matrix[3] = 0
    query = rng.normal(size=127).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1).astype(np.float32)
    expected_dot = matrix @ query
    expected_cosine = np.divide(
        expected_dot,
        norms,
        out=np.zeros_like(expected_dot),
        where=norms != 0,
    )
    assert np.allclose(dot_rows(matrix, query), expected_dot, atol=2e-5)
    assert np.allclose(
        cosine_rows(matrix, norms, query),
        expected_cosine,
        atol=2e-6,
    )


def test_partial_top_order_matches_stable_full_sort():
    rng = np.random.default_rng(16)
    scores = rng.integers(-20, 20, size=2000).astype(np.float32)
    assert np.array_equal(
        _top_order(scores, 37),
        np.argsort(-scores, kind="stable")[:37],
    )


def test_keyedvectors_pickle_roundtrip(vectors, tmp_path):
    ours, _ = vectors
    path = tmp_path / "vectors.kv"
    ours.save(path)
    loaded = KeyedVectors.load(path)
    assert loaded.index_to_key == ours.index_to_key
    assert np.array_equal(loaded.vectors, ours.vectors)


@pytest.mark.parametrize("binary", [False, True])
def test_word2vec_format_roundtrip(vectors, tmp_path, binary):
    ours, _ = vectors
    path = tmp_path / ("vectors.bin" if binary else "vectors.txt")
    ours.save_word2vec_format(path, binary=binary)
    loaded = KeyedVectors.load_word2vec_format(path, binary=binary)
    assert loaded.index_to_key == ours.index_to_key
    assert np.allclose(loaded.vectors, ours.vectors, atol=1e-7)


def test_word2vec_vocabulary_parity():
    corpus = [
        ["king", "queen", "royal"],
        ["queen", "royal"],
        ["dog", "pet"],
        ["king", "royal"],
    ]
    ours = Word2Vec(vector_size=10, min_count=1, seed=3)
    theirs = GWord2Vec(vector_size=10, min_count=1, seed=3, workers=1)
    ours.build_vocab(corpus)
    theirs.build_vocab(corpus)
    assert ours.wv.index_to_key == theirs.wv.index_to_key
    assert ours.corpus_count == theirs.corpus_count
    assert ours.corpus_total_words == theirs.corpus_total_words
    assert [
        ours.wv.get_vecattr(word, "count") for word in ours.wv.index_to_key
    ] == [
        theirs.wv.get_vecattr(word, "count") for word in theirs.wv.index_to_key
    ]


def _sgns_reference(inputs, outputs, centers, targets, negatives, alpha):
    inputs = inputs.copy()
    outputs = outputs.copy()
    for pair, (center, target) in enumerate(zip(centers, targets)):
        for sample, candidate in enumerate([target, *negatives[pair]]):
            label = float(sample == 0)
            score = np.dot(inputs[center], outputs[candidate])
            sigmoid = 1.0 / (1.0 + np.exp(-score))
            gradient = np.float32((label - sigmoid) * alpha)
            old_input = inputs[center].copy()
            old_output = outputs[candidate].copy()
            inputs[center] = old_input + gradient * old_output
            outputs[candidate] = old_output + gradient * old_input
    return inputs, outputs


def test_skipgram_kernel_matches_reference_update():
    rng = np.random.default_rng(8)
    inputs = rng.normal(scale=0.1, size=(5, 11)).astype(np.float32)
    outputs = rng.normal(scale=0.1, size=(5, 11)).astype(np.float32)
    centers = np.array([0, 2, 4], dtype=np.int64)
    targets = np.array([1, 3, 0], dtype=np.int64)
    negatives = np.array([[2, 4], [0, 1], [2, 3]], dtype=np.int64)
    expected_in, expected_out = _sgns_reference(
        inputs, outputs, centers, targets, negatives, 0.03
    )
    actual_in = inputs.copy()
    actual_out = outputs.copy()
    lib().mg_sgns_train(
        addr(actual_in, np.float32, writable=True),
        addr(actual_out, np.float32, writable=True),
        addr(centers, np.int64),
        addr(targets, np.int64),
        addr(negatives, np.int64),
        len(centers),
        inputs.shape[1],
        negatives.shape[1],
        0.03,
    )
    assert np.allclose(actual_in, expected_in, atol=2e-7)
    assert np.allclose(actual_out, expected_out, atol=2e-7)


def test_skipgram_pair_kernel_matches_dynamic_windows():
    tokens = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    indptr = np.array([0, 3, 5], dtype=np.int64)
    reduced = np.array([0, 1, 0, 0, 0], dtype=np.int64)
    centers = np.empty(20, dtype=np.int64)
    targets = np.empty(20, dtype=np.int64)
    count = lib().mg_sg_pairs(
        addr(tokens, np.int64),
        addr(indptr, np.int64),
        addr(reduced, np.int64),
        addr(centers, np.int64, writable=True),
        addr(targets, np.int64, writable=True),
        2,
        2,
    )
    expected = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (3, 4), (4, 3)]
    assert list(zip(centers[:count], targets[:count])) == expected


@pytest.mark.parametrize("sg", [0, 1])
def test_word2vec_training_is_deterministic_and_learns(sg):
    corpus = (
        [["king", "queen", "royal", "crown"]] * 100
        + [["dog", "cat", "pet", "animal"]] * 100
    )
    params = dict(
        vector_size=20,
        min_count=1,
        window=2,
        negative=5,
        epochs=15,
        alpha=0.05,
        min_alpha=0.01,
        sample=0,
        sg=sg,
        seed=2,
    )
    first = Word2Vec(corpus, **params)
    second = Word2Vec(corpus, **params)
    assert np.array_equal(first.wv.vectors, second.wv.vectors)
    related = max(
        first.wv.similarity("king", word) for word in ("queen", "royal", "crown")
    )
    unrelated = first.wv.similarity("king", "dog")
    assert related > unrelated + 0.2


def test_word2vec_constructor_signature_behavior():
    corpus = [["a", "b", "c"], ["a", "b"], ["x", "y"]] * 5
    model = Word2Vec(
        sentences=corpus,
        vector_size=8,
        min_count=1,
        workers=1,
        sg=1,
        negative=2,
        epochs=2,
    )
    assert model.wv.vector_size == 8
    assert model.corpus_count == len(corpus)
    assert np.isfinite(model.wv.vectors).all()
    prediction = model.predict_output_word(["a", "b"], topn=3)
    assert len(prediction) == 3
    assert sum(score for _, score in prediction) <= 1.0


@pytest.fixture
def tagged_docs():
    return [
        TaggedDocument(["red", "green", "color"], ["colors"]),
        TaggedDocument(["dog", "cat", "pet"], ["animals"]),
        TaggedDocument(["red", "blue", "color"], ["colors-two"]),
        TaggedDocument(["cat", "dog", "animal"], ["animals-two"]),
    ] * 10


@pytest.mark.parametrize("dm", [0, 1])
def test_doc2vec_training_and_inference(dm, tagged_docs):
    model = Doc2Vec(
        tagged_docs,
        vector_size=12,
        min_count=1,
        epochs=8,
        negative=3,
        dm=dm,
        seed=9,
    )
    assert model.dv.index_to_key == [
        "colors",
        "animals",
        "colors-two",
        "animals-two",
    ]
    first = model.infer_vector(["red", "color"], epochs=5)
    second = model.infer_vector(["red", "color"], epochs=5)
    assert np.array_equal(first, second)
    assert first.shape == (12,)
    assert np.isfinite(first).all()


def test_tagged_document_matches_upstream_container():
    from gensim.models.doc2vec import TaggedDocument as GTaggedDocument

    ours = TaggedDocument(["a", "b"], ["tag"])
    theirs = GTaggedDocument(["a", "b"], ["tag"])
    assert ours.words == theirs.words
    assert ours.tags == theirs.tags


@pytest.fixture
def bow_corpus():
    return [
        [(0, 1.0), (1, 2.0)],
        [(0, 2.0), (2, 1.0)],
        [(1, 1.0), (3, 3.0)],
        [(2, 2.0), (3, 1.0)],
        [(0, 1.0), (4, 2.0)],
        [(1, 2.0), (4, 1.0)],
    ]


def test_lsi_singular_values_and_projection_parity(bow_corpus):
    id2word = {index: str(index) for index in range(5)}
    ours = LsiModel(
        bow_corpus,
        num_topics=3,
        id2word=id2word,
        power_iters=2,
        extra_samples=20,
        random_seed=0,
    )
    theirs = GLsiModel(
        bow_corpus,
        num_topics=3,
        id2word=id2word,
        power_iters=2,
        extra_samples=20,
        random_seed=0,
    )
    assert np.allclose(ours.projection.s, theirs.projection.s, rtol=1e-10)
    for topic in range(3):
        a = ours.projection.u[:, topic]
        b = theirs.projection.u[:, topic]
        assert np.allclose(a, b, atol=1e-10) or np.allclose(a, -b, atol=1e-10)
    ours_coords = np.array([value for _, value in ours[bow_corpus[0]]])
    their_coords = np.array([value for _, value in theirs[bow_corpus[0]]])
    assert np.allclose(np.abs(ours_coords), np.abs(their_coords), atol=1e-10)


def test_lsi_corpus_transform_and_incremental_refit(bow_corpus):
    model = LsiModel(
        bow_corpus[:3],
        num_topics=2,
        id2word={index: str(index) for index in range(5)},
        random_seed=2,
    )
    model.add_documents(bow_corpus[3:])
    transformed = model[bow_corpus]
    assert len(transformed) == len(bow_corpus)
    assert model.docs_processed == len(bow_corpus)
    assert model.get_topics().shape == (2, 5)
    assert len(model.show_topic(0, topn=3)) == 3


def test_matrix_similarity_parity(bow_corpus):
    ours = MatrixSimilarity(bow_corpus, num_features=5)
    theirs = GMatrixSimilarity(bow_corpus, num_features=5)
    for query in bow_corpus:
        assert np.allclose(ours[query], theirs[query], atol=2e-7)


def test_empty_matrix_similarity():
    index = MatrixSimilarity([], num_features=4)
    assert len(index) == 0
    assert index[np.ones(4, dtype=np.float32)].shape == (0,)


@pytest.mark.parametrize("index_class", [MatrixSimilarity, SparseMatrixSimilarity])
def test_similarity_topn(index_class, bow_corpus):
    index = index_class(bow_corpus, num_features=5, num_best=3)
    result = index[bow_corpus[0]]
    expected_scores = GMatrixSimilarity(bow_corpus, num_features=5)[bow_corpus[0]]
    expected_order = np.argsort(-expected_scores, kind="stable")[:3]
    assert [doc for doc, _ in result] == expected_order.tolist()
    assert np.allclose(
        [score for _, score in result], expected_scores[expected_order], atol=2e-7
    )


def test_similarity_add_documents(tmp_path, bow_corpus):
    index = Similarity(tmp_path / "shard", bow_corpus[:3], 5, num_best=2)
    index.add_documents(bow_corpus[3:])
    assert len(index) == len(bow_corpus)
    assert index[bow_corpus[0]][0][0] == 0


def test_matutils_parity(bow_corpus):
    doc = bow_corpus[0]
    assert np.array_equal(matutils.sparse2full(doc, 5), gmat.sparse2full(doc, 5))
    dense = np.array([3.0, 0.0, 4.0])
    assert np.allclose(matutils.unitvec(dense), gmat.unitvec(dense))
    assert np.array_equal(
        matutils.corpus2dense(bow_corpus, 5),
        gmat.corpus2dense(bow_corpus, 5),
    )
    assert matutils.full2sparse(dense) == gmat.full2sparse(dense)
    assert matutils.dense2vec(dense) == gmat.dense2vec(dense)
    scores = np.array([2.0, -1.0, 3.0, 0.0])
    assert np.array_equal(
        matutils.argsort(scores, topn=2, reverse=True),
        gmat.argsort(scores, topn=2, reverse=True),
    )


def test_public_pickle_roundtrip(tmp_path, bow_corpus):
    model = LsiModel(bow_corpus, num_topics=2, random_seed=0)
    path = tmp_path / "lsi.pkl"
    model.save(path)
    loaded = LsiModel.load(path)
    assert np.array_equal(loaded.projection.u, model.projection.u)
    with path.open("rb") as stream:
        assert isinstance(pickle.load(stream), LsiModel)
