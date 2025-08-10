"""Model definitions for RC‑Mamba.

Currently this package exposes the `RCMamba` class in `rc_mamba.py`, which implements a retrieval‑conditioned Mamba block with projection‑level FiLM conditioning.  Additional models can be added here.
"""

from .rc_mamba import RCMamba, FiLMModulator

__all__ = ["RCMamba", "FiLMModulator"]