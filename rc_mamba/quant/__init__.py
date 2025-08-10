"""Quantization schemes for RC‑Mamba.

This package provides classes and functions for Finite Scalar Quantization (FSQ)
and k‑means based FSQ (KMeansFSQ).  The code here is designed to be plug‑and‑play
with PyTorch parameters.  It defines simple APIs for fitting a codebook to a
tensor and quantizing it at inference time.  A more complete implementation
would integrate these quantizers into the training loop and provide per‑channel
support.
"""

from .fsq import FSQ, KMeansFSQ

__all__ = ["FSQ", "KMeansFSQ"]