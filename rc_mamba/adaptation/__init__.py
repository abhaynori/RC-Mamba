"""Parameter adaptation mechanisms for RC‑Mamba.

This subpackage defines modules that adapt the model’s weights at inference time
based on retrieved information.  The current implementation provides a simple
`ParameterAdapter` that selects pre‑trained LoRA adapters keyed by topic IDs
returned from the retriever.  Researchers can extend this to implement
parameter hypernetworks, on‑the‑fly low‑rank updates, or other forms of dynamic
adaptation.
"""

from .param_adapter import ParameterAdapter

__all__ = ["ParameterAdapter"]