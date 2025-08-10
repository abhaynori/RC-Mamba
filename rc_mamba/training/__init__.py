"""Training routines for RC‑Mamba.

This package provides skeletons for supervised fine‑tuning (SFT), preference
optimisation (DPO), and LoRA fine‑tuning.  The implementations are meant as
guidelines; you should plug in your own dataset loaders, optimizers, and
accelerate/FSDP training loops as needed.  See the accompanying scripts for
examples of how to orchestrate training.
"""

from .sft import sft_train
from .dpo import dpo_train
from .lora import apply_lora

__all__ = ["sft_train", "dpo_train", "apply_lora"]