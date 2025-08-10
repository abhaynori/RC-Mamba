"""Gradio chat interface for RC‑Mamba.

This script provides a minimal Gradio interface that allows users to enter
queries, retrieve relevant contexts, and generate answers using the RC‑Mamba
model.  It illustrates how dynamic retrieval can be integrated into the
generation loop via a `RetrievalController`.  The app displays the current
retrieval passages alongside the model’s output.

Note: This UI is a stub.  To run it, install `gradio`, `sentence‑transformers`,
and other dependencies.  The retrieval and model inference functions are left
as placeholders; you should replace them with your own implementations.
"""

from __future__ import annotations

import gradio as gr
import torch

from typing import List, Tuple

from ..models.rc_mamba import RCMamba
from ..retrieval.index import RetrievalIndex
from ..retrieval.controller import RetrievalController


def launch_demo(model: RCMamba, retriever: RetrievalIndex, controller: RetrievalController, tokenizer) -> None:
    """Launch a Gradio interface for chat.

    Args:
        model: An instance of `RCMamba`.
        retriever: A retrieval index for context lookup.
        controller: A `RetrievalController` that decides when to refresh retrieval.
        tokenizer: A tokenizer with `encode` and `decode` methods.
    """

    def encode_query(query: str) -> torch.Tensor:
        return torch.tensor(tokenizer.encode(query, return_tensors="pt"))

    def retrieve_passages(query: str) -> List[str]:
        # Use the first embedder to vectorise the query; this is a placeholder
        query_vec = retriever.embedders[0](query)
        results = retriever.query(query_vec, k=3, mmr_lambda=0.2)
        passages = [retriever.documents[idx] for idx, _ in results]
        return passages

    def chat_fn(history: List[Tuple[str, str]], user_message: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        # Reset retrieval controller at the start of a new conversation
        if not history:
            controller.reset()
        # Retrieve passages and update retrieval embedding
        passages = retrieve_passages(user_message)
        # Convert passages to a single string for now (better would be to join with separators)
        retrieval_text = "\n".join(passages)
        retrieval_emb = torch.from_numpy(retriever.embedders[0](retrieval_text)).float().unsqueeze(0)
        model.update_retrieval(retrieval_emb)
        # Encode the chat context: concatenate previous turns and the user message
        context = "".join([f"User: {u}\nAssistant: {a}\n" for u, a in history]) + f"User: {user_message}\nAssistant:"
        input_ids = encode_query(context)
        # Generate a response with dynamic retrieval
        generated_ids = model.generate(
            input_ids.to(model.lm_head.weight.device),
            max_new_tokens=100,
            dynamic_retrieval_fn=controller,
        )
        reply = tokenizer.decode(generated_ids[0][input_ids.size(1):], skip_special_tokens=True)
        history.append((user_message, reply))
        return history, history

    with gr.Blocks() as demo:
        gr.Markdown("# RC‑Mamba Chat Demo (Experimental)")
        chat = gr.Chatbot()
        msg = gr.Textbox()
        state = gr.State([])
        msg.submit(chat_fn, [state, msg], [chat, state])
        msg.submit(lambda: "", None, msg)  # clear textbox
    demo.launch()


if __name__ == "__main__":
    # This part would normally load a pretrained model, tokenizer and
    # retrieval index.  Here we construct stubs so that the module can be run
    # without crashing.  Replace these with your actual model and retrieval setup.
    vocab_size = 1000
    model = RCMamba(vocab_size=vocab_size, d_model=64, n_layers=2, retrieval_dim=32)
    # Dummy tokenizer using a simple identity mapping
    class DummyTokenizer:
        def encode(self, text, return_tensors=None):
            # Split on spaces and map to indices modulo vocab_size
            ids = [hash(tok) % vocab_size for tok in text.split()]
            return torch.tensor([ids])
        def decode(self, ids, skip_special_tokens=True):
            return " ".join([str(i.item()) for i in ids])
    tokenizer = DummyTokenizer()
    # Dummy embedder: hash strings to random vectors in R^32
    import numpy as np
    def dummy_embed(text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.standard_normal(32)
    retriever = RetrievalIndex(embedders=[dummy_embed], documents=["Doc 1", "Doc 2", "Doc 3"], normalize=True)
    controller = RetrievalController(retrieve_fn=lambda q: dummy_embed(q), threshold=5.0, max_hops=2)
    launch_demo(model, retriever, controller, tokenizer)