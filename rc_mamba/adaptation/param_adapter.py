"""Parameter retrieval adaptation utilities.

This module implements a simple parameter adapter that selects or interpolates
between a set of pre‑trained LoRA adapters based on a retrieved topic vector.
The idea is to map high‑level topics (or clusters) of queries to sets of
low‑rank updates that specialize the base model on demand.  A practical
implementation would pre‑compute LoRA weight deltas for a range of topics,
store them in a lookup table, and perform a convex combination of these
updates during inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List


class ParameterAdapter(nn.Module):
    """Simple LoRA selection and interpolation based on retrieved topics.

    The adapter holds a dictionary mapping string topic IDs to LoRA update
    tensors (one per trainable parameter).  When called with a topic weight
    vector, it computes a weighted sum of the corresponding LoRA updates and
    applies them to the base model’s parameters.  This class does not manage
    the base model; you should update the base model’s parameters externally
    using the returned deltas.
    """

    def __init__(self, topic_loras: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """Initialize with a dictionary of LoRA weight deltas.

        Args:
            topic_loras: A mapping from topic strings to another mapping of
                parameter names to low‑rank update tensors.  For example,
                `topic_loras['sports']['lm_head.weight']` could be a tensor of
                shape `(vocab_size, d_model)` representing the LoRA delta for
                the `lm_head.weight` of an RCMamba model.
        """
        super().__init__()
        self.topic_loras = topic_loras
        # Register the LoRA deltas as buffers so they move with the adapter
        for topic, deltas in topic_loras.items():
            for name, tensor in deltas.items():
                self.register_buffer(f"{topic}.{name}", tensor)

    def forward(self, topic_weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Compute a convex combination of LoRA updates.

        Args:
            topic_weights: Mapping from topic strings to scalar weights.  The
                weights should sum to 1 to form a convex combination.

        Returns:
            A dictionary mapping parameter names to the interpolated LoRA delta.
        """
        # Initialize an empty dictionary of accumulated deltas
        accumulated: Dict[str, torch.Tensor] = {}
        for topic, weight in topic_weights.items():
            if weight <= 0 or topic not in self.topic_loras:
                continue
            for name, _ in self.topic_loras[topic].items():
                delta = getattr(self, f"{topic}.{name}")
                if name not in accumulated:
                    accumulated[name] = weight * delta
                else:
                    accumulated[name] = accumulated[name] + weight * delta
        return accumulated