"""A Mojo-backed subset of gensim."""

from . import matutils, models, similarities
from .models import Doc2Vec, KeyedVectors, LsiModel, TaggedDocument, Word2Vec

__version__ = "0.1.0"

__all__ = [
    "Doc2Vec",
    "KeyedVectors",
    "LsiModel",
    "TaggedDocument",
    "Word2Vec",
    "matutils",
    "models",
    "similarities",
]
