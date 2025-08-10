"""LoRA utilities for RC‑Mamba.

This module provides a simple function to add LoRA adapters to a linear
projection in an RC‑Mamba model.  It does not rely on the PEFT library and
instead demonstrates how to manually construct low‑rank adapters.  For a more
complete solution, consider using PEFT's LoRA implementation.
"""

import torch
import torch.nn as nn
from typing import Tuple


def apply_lora(
    linear: nn.Linear,
    rank: int,
    scaling: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """Attach a LoRA adapter to a linear layer.

    This function creates two low‑rank matrices `A` and `B` such that
    `delta_W = (A @ B) * scaling` is added to the original weight matrix of
    the linear layer.  It returns the modified linear layer, the two LoRA
    parameters, and the original weight.  The LoRA parameters are created as
    trainable parameters, while the original weight is frozen.

    Args:
        linear: A `nn.Linear` module to which the adapter will be attached.
        rank: Rank of the low‑rank update (typically small, e.g. 4 or 8).
        scaling: Scaling factor applied to the LoRA update during the forward pass.
        device: Device on which to allocate the LoRA parameters.

    Returns:
        A tuple `(wrapped_linear, A, B)` where `wrapped_linear` is a new module
        that applies the LoRA update during its forward pass, and `A` and `B`
        are the trainable low‑rank matrices.
    """
    # Freeze the original weights
    for param in linear.parameters():
        param.requires_grad = False
    in_dim, out_dim = linear.in_features, linear.out_features
    # Create low‑rank matrices
    A = nn.Parameter(torch.zeros((rank, in_dim), device=device))
    B = nn.Parameter(torch.zeros((out_dim, rank), device=device))
    # Initialize A and B
    nn.init.kaiming_uniform_(A, a=math.sqrt(5))
    nn.init.zeros_(B)

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, A: nn.Parameter, B: nn.Parameter, scaling: float):
            super().__init__()
            self.base = base
            self.A = A
            self.B = B
            self.scaling = scaling

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            delta_weight = self.B @ self.A  # (out_dim, in_dim)
            return F.linear(x, self.base.weight + self.scaling * delta_weight, self.base.bias)

    wrapped = LoRALinear(linear, A, B, scaling)
    return wrapped, A, B