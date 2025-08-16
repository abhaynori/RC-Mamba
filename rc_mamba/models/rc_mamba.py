"""Retrieval‑Conditioned Mamba model definitions.

This module defines the `RCMamba` model, which extends the Mamba state space model by
introducing **projection‑level FiLM** conditioning on the state space matrices
\(B\) and \(C\) using an external retrieval embedding.  The goal is to allow
external knowledge retrieved from a corpus to modulate the dynamics of the hidden
state without resorting to attention.  A `FiLMModulator` network transforms
retrieval embeddings into scale and shift vectors for the B and C projections.

The implementation is intentionally high‑level and does not depend on the
`mamba_ssm` library.  It is written in a way that should be familiar to
researchers and can be extended with custom Mamba blocks or third‑party SSM
implementations.  Users should fill in the TODOs to integrate a real Mamba
backbone such as the one provided by `mamba_ssm.ops` or the `state‑spaces`
repository.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from .mamba_ssm import MambaBlock


class FiLMModulator(nn.Module):
    """Maps a retrieval embedding to FiLM parameters for the B and C projections.

    Given a retrieval embedding `r` of shape `(batch, d_r)`, this module outputs
    four tensors `(gamma_B, beta_B, gamma_C, beta_C)` each of shape `(d_proj,)`
    that are used to scale and shift the B and C matrices of a Mamba block:

    .. math::

       \tilde{B} = (1 + \gamma_B) \odot B + \beta_B,\\
       \tilde{C} = (1 + \gamma_C) \odot C + \beta_C.

    The architecture is intentionally simple: a two‑layer MLP with SiLU
    activation, mapping `d_r` inputs to `4*d_proj` outputs.  Researchers can
    experiment with deeper networks or attention mechanisms here.
    """

    def __init__(self, retrieval_dim: int, proj_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = proj_dim * 2
        self.fc1 = nn.Linear(retrieval_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 4 * proj_dim)

    def forward(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # r: (batch, d_r)
        x = F.silu(self.fc1(r))
        params = self.fc2(x)  # (batch, 4*proj_dim)
        gamma_B, beta_B, gamma_C, beta_C = params.chunk(4, dim=-1)
        return gamma_B, beta_B, gamma_C, beta_C


class RCMambaBlock(nn.Module):
    """A single RCMamba block.

    This block wraps a real Mamba SSM layer and applies FiLM
    conditioning to its B and C projection matrices.
    """

    def __init__(self, d_model: int, d_state: int, expand: int, retrieval_dim: int):
        super().__init__()
        # Real Mamba SSM implementation
        self.ssm = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            expand=expand
        )
        self.d_model = d_model
        self.d_state = d_state
        # FiLM modulator maps retrieval embeddings to FiLM parameters
        self.modulator = FiLMModulator(retrieval_dim=retrieval_dim, proj_dim=d_state)

    def forward(self, x: torch.Tensor, retrieval: torch.Tensor) -> torch.Tensor:
        """Forward pass with FiLM conditioning.

        Args:
            x: Token representations of shape `(batch, seq_len, d_model)`.
            retrieval: Retrieval embedding of shape `(batch, d_r)`.

        Returns:
            Updated token representations with shape `(batch, seq_len, d_model)`.
        """
        # Get FiLM parameters from retrieval embedding
        gamma_B, beta_B, gamma_C, beta_C = self.modulator(retrieval)
        film_params = (gamma_B, beta_B, gamma_C, beta_C)
        
        # Apply FiLM-conditioned Mamba block
        return self.ssm(x, film_params=film_params)


class RCMamba(nn.Module):
    """Retrieval‑Conditioned Mamba model.

    This model stacks multiple `RCMambaBlock`s and includes token embedding and
    output projection layers.  It supports generation with optional state
    caching and dynamic retrieval updating.  The core logic of Mamba is left
    abstract; you need to integrate a real state space implementation.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 8,
        d_state: int = 16,
        expand: int = 2,
        retrieval_dim: int = 256,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            RCMambaBlock(d_model=d_model, d_state=d_state, expand=expand, retrieval_dim=retrieval_dim)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.retrieval_dim = retrieval_dim
        # A buffer to hold the current retrieval embedding; can be updated by the dynamic retrieval controller
        self.register_buffer("current_retrieval", torch.zeros(1, retrieval_dim))

    def forward(self, input_ids: torch.Tensor, retrieval: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Look up embeddings
        x = self.embed(input_ids)  # (batch, seq_len, d_model)
        # Use provided retrieval embedding or fall back to cached
        if retrieval is None:
            retrieval = self.current_retrieval.expand(x.size(0), -1)
        # Pass through FiLM‑conditioned Mamba blocks
        for layer in self.layers:
            x = layer(x, retrieval)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        eos_token_id: Optional[int] = None,
        retrieval: Optional[torch.Tensor] = None,
        dynamic_retrieval_fn: Optional[callable] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with optional dynamic retrieval.

        Args:
            input_ids: Input token IDs `(1, seq_len)`.
            max_new_tokens: Maximum number of new tokens to generate.
            eos_token_id: Optional end‑of‑sequence token ID to stop early.
            retrieval: Initial retrieval embedding `(1, d_r)`; if None, uses cached.
            dynamic_retrieval_fn: Optional function taking `(generated_ids, logits)`
                and returning a new retrieval embedding.  If provided, this function
                is invoked every few steps to decide whether to refresh retrieval.

        Returns:
            Token IDs including the newly generated tokens.
        """
        if retrieval is None:
            retrieval = self.current_retrieval
        generated = input_ids
        states: Optional[List[torch.Tensor]] = None
        for _ in range(max_new_tokens):
            logits = self.forward(generated, retrieval=retrieval)  # (1, seq_len, vocab)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            # Check EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            # Optionally update retrieval based on dynamic retrieval function
            if dynamic_retrieval_fn is not None:
                retrieval = dynamic_retrieval_fn(generated, next_token_logits)
        return generated

    def update_retrieval(self, new_retrieval: torch.Tensor) -> None:
        """Update the cached retrieval embedding.

        This method can be called externally (e.g. by a retrieval controller) to
        refresh the retrieval embedding used during forward and generation.
        """
        self.current_retrieval = new_retrieval.detach()