"""Direct Preference Optimisation (DPO) training for RC‑Mamba.

This module implements a skeleton of the DPO objective used to align models
based on preference data.  It includes an optional per‑example interpolation
between supervised fine‑tuning (SFT) and DPO losses, controlled by an
uncertainty score (π‑DPO).  A typical batch of preference data consists of
tuples `(prompt, chosen, rejected)`, and the corresponding model reference
scores come from a frozen reference model.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer
from typing import Iterable, Dict, Callable, Optional


def dpo_train(
    model: nn.Module,
    ref_model: nn.Module,
    data_loader: Iterable[Dict[str, torch.Tensor]],
    optimizer: Optimizer,
    beta: float = 0.1,
    alpha: float = 2.0,
    uncertainty_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    pad_token_id: int = 0,
) -> None:
    """Train the model using the DPO objective with optional π‑DPO scheduling.

    Args:
        model: The trainable RC‑Mamba model (policy).
        ref_model: A frozen reference model providing baseline probabilities.
        data_loader: An iterable of batches containing `input_ids`,
            `chosen_ids`, and `rejected_ids` token ID tensors.
        optimizer: Optimizer for updating the model parameters.
        beta: Temperature parameter for the DPO loss.
        alpha: Scaling factor used in the uncertainty gating of π‑DPO.
        uncertainty_fn: Optional function computing per‑example uncertainty
            scores from the model and reference logits.  If provided, it
            determines the mixture weight between SFT and DPO losses.
        pad_token_id: Token ID used for padding.
    """
    model.train()
    ref_model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    for batch in data_loader:
        input_ids = batch["input_ids"]  # (batch, seq_len)
        chosen_ids = batch["chosen_ids"]  # (batch, seq_len)
        rejected_ids = batch["rejected_ids"]  # (batch, seq_len)
        # Compute policy and reference logits
        policy_logits = model(input_ids)
        ref_logits = ref_model(input_ids).detach()
        # Compute per‑token log probabilities
        policy_logp = torch.log_softmax(policy_logits, dim=-1)
        ref_logp = torch.log_softmax(ref_logits, dim=-1)
        # Gather log probabilities of chosen and rejected sequences
        def gather_logp(logp, seq):
            return logp.gather(2, seq.unsqueeze(-1)).squeeze(-1)
        logp_chosen = gather_logp(policy_logp, chosen_ids)
        logp_rejected = gather_logp(policy_logp, rejected_ids)
        ref_logp_chosen = gather_logp(ref_logp, chosen_ids)
        ref_logp_rejected = gather_logp(ref_logp, rejected_ids)
        # Sum log probabilities over sequence length
        policy_chosen_sum = logp_chosen.sum(dim=-1)
        policy_rejected_sum = logp_rejected.sum(dim=-1)
        ref_chosen_sum = ref_logp_chosen.sum(dim=-1)
        ref_rejected_sum = ref_logp_rejected.sum(dim=-1)
        # Compute the DPO loss per example
        delta = policy_chosen_sum - policy_rejected_sum - (ref_chosen_sum - ref_rejected_sum)
        dpo_losses = -torch.log(torch.sigmoid(beta * delta))
        # Optional π‑DPO: interpolate with SFT loss based on uncertainty
        if uncertainty_fn is not None:
            # Compute an uncertainty score u in [0, 1] per example
            u = uncertainty_fn(policy_logits.detach(), ref_logits)
            # Compute the SFT loss on chosen sequences only
            sft_loss = loss_fn(policy_logits.view(-1, policy_logits.size(-1)), chosen_ids.view(-1))
            # Blend SFT and DPO using a sigmoid of scaled uncertainty
            lam = torch.sigmoid(alpha * (u - 0.5))  # higher u ⇒ more DPO
            total_loss = (1 - lam) * sft_loss + lam * dpo_losses.mean()
        else:
            total_loss = dpo_losses.mean()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()