"""Supervised fine‑tuning (SFT) loop for RC‑Mamba.

This function illustrates how to perform supervised fine‑tuning of an RC‑Mamba
model on a text dataset.  It is intentionally high‑level and omits many
details (e.g. batching, accelerator support, dataset streaming).  See the
docstring for guidance on integrating your own data pipeline and using
accelerate or DeepSpeed for scaling.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer
from typing import Callable, Iterable, Dict


def sft_train(
    model: nn.Module,
    data_loader: Iterable[Dict[str, torch.Tensor]],
    optimizer: Optimizer,
    tokenizer,
    epochs: int = 1,
    pad_token_id: int = 0,
) -> None:
    """Perform supervised fine‑tuning on the provided data loader.

    Each item in the data loader should be a dictionary containing at least
    `input_ids` and optionally `labels`.  If `labels` is not provided, the
    labels are set equal to the input IDs (next‑token prediction).  A simple
    cross‑entropy loss is used.  This function does not handle gradient
    accumulation or distributed training; you should wrap it with your own
    accelerator if necessary.
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    for epoch in range(epochs):
        for batch in data_loader:
            input_ids = batch["input_ids"]  # (batch, seq_len)
            labels = batch.get("labels", input_ids.clone())
            logits = model(input_ids)
            # Shift so that tokens predict the next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()