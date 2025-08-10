"""Evaluation metrics for RC‑Mamba.

This module defines simple functions for computing perplexity and long‑context
recall.  The long‑context recall metric measures how well a model can recall a
token from the beginning of a long sequence.  Users can extend these
functions to include other metrics such as BLEU, ROUGE, and factual recall.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F
from typing import Dict, Iterable


@torch.no_grad()
def perplexity(model, data_loader: Iterable[Dict[str, torch.Tensor]], pad_token_id: int = 0) -> float:
    """Compute the perplexity of `model` over `data_loader`.

    Args:
        model: A language model that accepts `input_ids` and returns logits.
        data_loader: Iterable of batches containing `input_ids`.
        pad_token_id: Token ID used for padding; ignored in the loss.

    Returns:
        The geometric mean perplexity over the dataset.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in data_loader:
        input_ids = batch["input_ids"]
        logits = model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=pad_token_id,
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += (shift_labels != pad_token_id).sum().item()
    return float(torch.exp(torch.tensor(total_loss / total_tokens)))


@torch.no_grad()
def long_context_recall(model, tokenizer, seq_len: int = 4096, num_samples: int = 10) -> float:
    """Evaluate long‑context recall on synthetic sequences.

    We generate synthetic sequences of length `seq_len` consisting of random
    tokens.  The first token is designated as the key.  The model is asked to
    predict the key token at the end of the sequence.  The recall is the
    fraction of times the model predicts the correct key.

    Args:
        model: A language model with a `.generate()` method.
        tokenizer: Tokenizer used to decode and encode tokens.
        seq_len: Length of the synthetic sequence.
        num_samples: Number of sequences to evaluate.

    Returns:
        The average recall over `num_samples` sequences.
    """
    recall = 0.0
    vocab_size = model.lm_head.out_features
    for _ in range(num_samples):
        # Generate a random key and random sequence
        key_id = torch.randint(0, vocab_size, (1, 1))
        body = torch.randint(0, vocab_size, (1, seq_len - 2))
        input_ids = torch.cat([key_id, body, key_id], dim=1)
        # Ask the model to generate one token at the end
        generated = model.generate(input_ids[:, :-1], max_new_tokens=1)
        pred_id = generated[0, -1]
        recall += float(pred_id == key_id)
    return recall / num_samples