# RC‑Mamba Expanded Project

This project is a research‑grade prototype for **Retrieval‑Conditioned Mamba** (RC‑Mamba), a state‑space model (SSM) that integrates external retrieval signals directly into the hidden dynamics of the Mamba architecture.  It builds on our previous prototype by adding several novel components designed to push the boundaries of retrieval‑augmented state‑space models and explore new research questions suitable for NeurIPS‑level submissions.

## Highlights

* **Projection‑level FiLM conditioning:** Retrieval embeddings modulate the Mamba SSM matrices \(B\) and \(C\) via Feature‑wise Linear Modulation (FiLM) gates.  This allows the hidden state dynamics to depend on external evidence without adding attention layers.
* **Dynamic multi‑hop retrieval:** A controller monitors model uncertainty and triggers additional retrieval hops when needed.  The retrieval state is refreshed on‑demand, enabling multi‑hop reasoning over large corpora.
* **Cross‑modal and cross‑lingual retrieval:** The retrieval pipeline supports embeddings from different modalities (text, images, audio) and multiple languages through a shared projection layer.
* **Parameter retrieval adaptation:** The model can dynamically select LoRA adapters based on retrieved topics, enabling lightweight on‑the‑fly adaptation of the underlying weights.
* **Dual‑codebook quantization:** Both uniform Finite Scalar Quantization (FSQ) and per‑channel k‑means FSQ are available.  An adaptive policy can choose bit‑widths on the fly based on model uncertainty.
* **π‑DPO training schedule:** A single‑loop Direct Preference Optimisation loss with per‑example interpolation between supervised fine‑tuning (SFT) and DPO, controlled by an uncertainty measure.
* **Retrieval‑augmented evaluation:** Evaluation utilities include long‑context recall tasks, memory vs. retrieval K sweeps, latency vs. quantization level plots, and a synthetic needle‑in‑a‑haystack benchmark.

This repository is meant as a starting point for researchers.  Many modules provide skeleton implementations and detailed docstrings to guide further development.  You are encouraged to fill in the implementations, experiment with new conditioning mechanisms, and extend the evaluation suite.

## Usage

To install the package in editable mode and run a quick demo:

```bash
pip install -e .
python -m rc_mamba.ui.app
```

Full training, DPO tuning, hyperparameter sweeps, and paper generation are orchestrated via the scripts in the `scripts/` directory.  See the `scripts/README.md` for details.