"""Finite Scalar Quantization (FSQ) and K‑Means FSQ for RC‑Mamba.

This module implements simple uniform and k‑means based quantization schemes
for linear layer weights.  Both quantizers operate on a single tensor at a
time and return a quantized tensor along with any codebook indices.  These
implementations are minimal and meant for experimentation; they do not support
fast inference kernels out of the box.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FSQ:
    """Uniform Finite Scalar Quantizer.

    This quantizer maps a floating‑point tensor to a finite set of equally spaced
    levels between the minimum and maximum values of the tensor.  The codebook
    can be shared across tensors or computed per‑tensor on demand.
    """

    def __init__(self, levels: int = 16):
        if levels < 2:
            raise ValueError("levels must be >= 2")
        self.levels = levels

    def fit(self, tensor: torch.Tensor) -> None:
        """Compute the codebook parameters from a tensor.

        Args:
            tensor: Floating‑point tensor to quantize.
        """
        self.min_val = tensor.min().item()
        self.max_val = tensor.max().item()
        # Avoid degenerate range
        if self.max_val - self.min_val < 1e-8:
            self.delta = 1e-8
        else:
            self.delta = (self.max_val - self.min_val) / (self.levels - 1)

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a tensor using the current codebook.

        Returns:
            (q, indices) where q is the quantized tensor and indices are the
            integer codebook indices.
        """
        if not hasattr(self, "delta"):
            self.fit(tensor)
        indices = torch.round((tensor - self.min_val) / self.delta).clamp(0, self.levels - 1)
        quantized = indices * self.delta + self.min_val
        return quantized, indices.to(torch.int64)


class KMeansFSQ:
    """K‑means Finite Scalar Quantizer.

    This quantizer uses k‑means to learn a set of codebook values for a tensor
    (optionally per output channel).  It is useful when the distribution of
    weights is highly non‑uniform.  The fitting procedure is simplified and
    non‑iterative; for more accurate codebooks, consider running iterative k‑means.
    """

    def __init__(self, levels: int = 16, per_channel: bool = True, max_iter: int = 10):
        self.levels = levels
        self.per_channel = per_channel
        self.max_iter = max_iter

    def fit(self, tensor: torch.Tensor) -> None:
        """Compute k‑means codebooks for the given tensor.

        If `per_channel` is True, a separate codebook is fitted for each output
        channel (dimension 0).  Otherwise a global codebook is used.
        """
        if self.per_channel:
            # Reshape to (out_channels, -1)
            flat = tensor.reshape(tensor.size(0), -1)
            self.codebooks = []
            for row in flat:
                codebook = self._kmeans_1d(row, self.levels, self.max_iter)
                self.codebooks.append(codebook)
        else:
            flat = tensor.view(-1)
            self.codebooks = self._kmeans_1d(flat, self.levels, self.max_iter)

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a tensor using the learned codebooks.

        Returns:
            (q, indices) where q is the quantized tensor and indices are the
            integer codebook indices.  If `per_channel` is True, indices has
            shape `(out_channels, *)` matching the first dimension of `tensor`.
        """
        if not hasattr(self, "codebooks"):
            self.fit(tensor)
        if self.per_channel:
            quantized = []
            indices = []
            for i, row in enumerate(tensor.reshape(tensor.size(0), -1)):
                cb = self.codebooks[i]
                dist = (row.unsqueeze(1) - cb.unsqueeze(0)).abs()
                idx = torch.argmin(dist, dim=1)
                qrow = cb[idx]
                quantized.append(qrow)
                indices.append(idx)
            q_tensor = torch.stack(quantized).reshape_as(tensor)
            idx_tensor = torch.stack(indices).reshape_as(tensor)
        else:
            flat = tensor.view(-1)
            cb = self.codebooks
            dist = (flat.unsqueeze(1) - cb.unsqueeze(0)).abs()
            idx = torch.argmin(dist, dim=1)
            qflat = cb[idx]
            q_tensor = qflat.view_as(tensor)
            idx_tensor = idx.view_as(tensor)
        return q_tensor, idx_tensor

    def _kmeans_1d(self, data: torch.Tensor, k: int, max_iter: int) -> torch.Tensor:
        """Simple 1‑D k‑means for initializing codebooks.

        Args:
            data: 1‑D tensor of samples.
            k: Number of clusters.
            max_iter: Maximum number of k‑means iterations.

        Returns:
            Tensor of shape `(k,)` with cluster centers.
        """
        # Initialize codebook with linear spacing between min and max
        min_val, max_val = data.min(), data.max()
        codebook = torch.linspace(min_val, max_val, k, device=data.device)
        for _ in range(max_iter):
            # Assign points to nearest centroid
            distances = (data.unsqueeze(1) - codebook.unsqueeze(0)).abs()
            assignments = torch.argmin(distances, dim=1)
            # Update centroids
            new_codebook = []
            for j in range(k):
                mask = assignments == j
                if mask.any():
                    new_codebook.append(data[mask].mean())
                else:
                    # Keep existing centroid if no points assigned
                    new_codebook.append(codebook[j])
            new_codebook = torch.stack(new_codebook)
            # If codebook didn't change, stop early
            if torch.allclose(new_codebook, codebook):
                break
            codebook = new_codebook
        return codebook