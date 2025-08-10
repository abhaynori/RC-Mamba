"""Retrieval utilities for RC‑Mamba.

The retrieval subpackage provides building blocks for constructing retrieval‑
augmented pipelines.  It includes an `Index` class for building and querying
FAISS or other vector indices, a `retrieve` function that returns relevant
passages or embeddings for a query, and a `RetrievalController` that decides
when to refresh retrieval based on model outputs.  Cross‑modal retrieval is
supported via a common projection layer; you can plug in CLIP or audio
embedders to generate embeddings for non‑text data.
"""

from .index import RetrievalIndex
from .controller import RetrievalController

__all__ = ["RetrievalIndex", "RetrievalController"]