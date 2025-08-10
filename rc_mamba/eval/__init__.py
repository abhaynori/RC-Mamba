"""Evaluation utilities for RC‑Mamba.

This subpackage provides functions for calculating perplexity, long‑context
recall metrics, and other evaluation metrics specific to retrieval‑augmented
models.  It also includes synthetic tasks for testing memory and retrieval
capacity, such as needle‑in‑a‑haystack retrieval and key‑value recall.
"""

from .metrics import perplexity, long_context_recall

__all__ = ["perplexity", "long_context_recall"]