"""Vector retrieval index with MMR and cross‑modal support.

This module implements a generic retrieval index that can store and retrieve
vector representations of documents, images or other modalities.  It provides
a simple API for building an index from a corpus, performing queries, and
computing Maximal Marginal Relevance (MMR) rankings to increase diversity.
Cross‑modal retrieval is achieved by passing a list of embedders, each of
which maps raw documents (text strings, image paths, etc.) into a common
embedding space.

For actual deployment, install `sentence‑transformers`, `faiss‑cpu` (or
`faiss‑gpu`), and optionally other modality embedders like `CLIP`.  In this
prototype, methods are left as stubs or simple vector operations to keep
dependencies minimal.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, List, Tuple, Optional, Sequence, Any


class RetrievalIndex:
    """A minimal retrieval index with optional MMR diversification.

    Arguments:
        embedders: List of callables taking a raw document and returning a
            vector representation (numpy array).  At least one embedder must
            be provided.  Multiple embedders enable cross‑modal retrieval.
        documents: Sequence of raw documents (strings, file paths, etc.).  The
            order of documents determines their integer IDs in the index.
        normalize: Whether to L2‑normalize embeddings before indexing.

    Attributes:
        index (np.ndarray): Matrix of shape `(n_docs, d)` containing all
            embeddings.  The dimension `d` is determined by the first embedder.
    """

    def __init__(
        self,
        embedders: Sequence[Callable[[Any], np.ndarray]],
        documents: Sequence[Any],
        normalize: bool = True,
    ) -> None:
        if not embedders:
            raise ValueError("At least one embedder must be provided.")
        self.embedders = embedders
        self.normalize = normalize
        # Compute embeddings for all documents using the first embedder.  You
        # could extend this to multi‑modal fusion (e.g. concatenate or sum
        # embeddings), but for simplicity we just use the first one here.
        vecs = []
        for doc in documents:
            v = embedders[0](doc)
            vecs.append(v)
        self.index = np.stack(vecs)
        if normalize:
            self.index = self.index / (np.linalg.norm(self.index, axis=1, keepdims=True) + 1e-8)
        self.documents = list(documents)

    def query(
        self, query_vec: np.ndarray, k: int = 5, mmr_lambda: float = 0.0
    ) -> List[Tuple[int, float]]:
        """Query the index and return document IDs and similarity scores.

        Args:
            query_vec: Query embedding (numpy array).
            k: Number of results to return.
            mmr_lambda: If >0, perform Maximal Marginal Relevance (MMR)
                diversification with this weight (0 means no diversification).

        Returns:
            List of `(doc_id, score)` tuples sorted by decreasing score.
        """
        q = query_vec
        if self.normalize:
            q = q / (np.linalg.norm(q) + 1e-8)
        scores = self.index @ q  # cosine similarity if normalized
        candidate_ids = np.argsort(-scores)[:k * 4]  # take more for MMR pool
        selected = []
        selected_scores = []
        for idx in candidate_ids:
            if len(selected) >= k:
                break
            s = scores[idx]
            if mmr_lambda > 0 and selected:
                # Compute redundancy penalty
                redund = max(np.dot(self.index[idx], self.index[sel]) for sel in selected)
                mmr_score = mmr_lambda * s - (1 - mmr_lambda) * redund
            else:
                mmr_score = s
            selected.append(idx)
            selected_scores.append(mmr_score)
        # Sort selected by final MMR scores
        pairs = list(zip(selected, selected_scores))
        pairs.sort(key=lambda x: -x[1])
        return pairs[:k]

    def __len__(self) -> int:
        return len(self.documents)