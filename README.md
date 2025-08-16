# RC-Mamba: Retrieval-Conditioned State Space Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive research framework for **Retrieval-Conditioned Mamba** (RC-Mamba), implementing state-space models with external retrieval mechanisms for enhanced reasoning across long-context, multimodal, and cross-lingual tasks.

## 🚀 Overview

RC-Mamba represents a novel architecture that integrates external retrieval signals directly into the hidden dynamics of Mamba state-space models. This enables efficient processing of long sequences while maintaining access to external knowledge through retrieval-augmented mechanisms.

### Key Innovations

- **🎛️ FiLM Conditioning**: Feature-wise Linear Modulation of SSM parameters based on retrieval embeddings
- **🔄 Dynamic Multi-hop Retrieval**: Uncertainty-driven adaptive retrieval for complex reasoning tasks
- **🌍 Cross-modal Support**: Unified retrieval across text, image, and audio modalities
- **📊 Dual-Codebook Quantization**: Adaptive compression with uniform and k-means finite scalar quantization
- **🎯 π-DPO Training**: Advanced preference optimization with uncertainty-based SFT/DPO mixing
- **📏 Comprehensive Evaluation**: Multi-dimensional benchmarking across diverse tasks

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Experiments](#experiments)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [Citation](#citation)

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (for GPU acceleration)
- Git LFS (for large model files)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/your-username/RC-Mamba.git
cd RC-Mamba

# Install in development mode
pip install -e .

# Or install research dependencies
pip install -r requirements-research.txt
```

### Docker Installation

```bash
# Build Docker image
docker build -t rc-mamba .

# Run container
docker run --gpus all -it rc-mamba
```

## 🚀 Quick Start

### Basic Usage

```python
import torch
from rc_mamba.models import RCMambaModel
from rc_mamba.retrieval import MultiModalRetriever

# Initialize model
model = RCMambaModel(
    vocab_size=32000,
    d_model=768,
    n_layers=12,
    retrieval_dim=512
)

# Initialize retriever
retriever = MultiModalRetriever(
    text_model="sentence-transformers/all-MiniLM-L6-v2",
    image_model="openai/clip-vit-base-patch32",
    embedding_dim=512
)

# Process input with retrieval
input_ids = torch.randint(0, 32000, (1, 128))
retrieved_embeddings = retriever.retrieve_text("What is machine learning?", k=5)

# Forward pass
outputs = model(input_ids, retrieval_embeddings=retrieved_embeddings)
print(f"Output shape: {outputs.logits.shape}")
```

### Training Example

```python
from rc_mamba.training import PiDPOTrainer
from rc_mamba.data import DatasetFactory

# Create datasets
train_dataset = DatasetFactory.create_dataset("narrativeqa", split="train")
eval_dataset = DatasetFactory.create_dataset("narrativeqa", split="validation")

# Initialize trainer
trainer = PiDPOTrainer(
    model=model,
    retriever=retriever,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    sft_weight=0.7,
    dpo_weight=0.3
)

# Train model
trainer.train(num_epochs=3, save_dir="checkpoints/")
```

## 🏗️ Architecture

### Core Components

#### 1. Mamba SSM with FiLM Conditioning
```python
class MambaSSM(nn.Module):
    """
    Selective State Space Model with FiLM conditioning for retrieval integration.
    
    Key features:
    - Selective state space mechanism
    - FiLM modulation of B and C matrices
    - Efficient parallel and recurrent computation modes
    """
```

#### 2. Multi-Modal Retrieval System
```python
class MultiModalRetriever:
    """
    Cross-modal retrieval system supporting text, image, and audio.
    
    Components:
    - Unified embedding space
    - FAISS-based similarity search
    - Multi-hop retrieval with uncertainty triggering
    """
```

#### 3. Dual-Codebook Quantization
```python
class DualCodebookQuantizer:
    """
    Adaptive quantization system with dual codebooks.
    
    Features:
    - Uniform and k-means finite scalar quantization
    - Dynamic bit-width selection
    - Compression ratio analysis
    """
```

### Model Configuration

```yaml
# config/model_config.yaml
model:
  vocab_size: 32000
  d_model: 768
  n_layers: 12
  ssm:
    d_state: 16
    d_conv: 4
    expand_factor: 2
  retrieval:
    embedding_dim: 512
    max_retrieved: 10
    film_conditioning: true
```

## 🎓 Training

### Standard Training

```bash
# Basic training
python scripts/train_rc_mamba.py \
  --config config/train_config.yaml \
  --dataset narrativeqa \
  --model_size base \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --num_epochs 5

# With distributed training
torchrun --nproc_per_node=4 scripts/train_rc_mamba.py \
  --config config/train_config.yaml \
  --distributed
```

### π-DPO Training

```bash
# Preference optimization training
python scripts/train_rc_mamba.py \
  --training_method pi_dpo \
  --sft_weight 0.7 \
  --dpo_weight 0.3 \
  --uncertainty_threshold 0.5 \
  --preference_dataset anthropic_hh
```

### Advanced Training Options

```bash
# With quantization
python scripts/train_rc_mamba.py \
  --use_quantization \
  --quantization_method dual_codebook \
  --compression_ratio 4.0

# With LoRA adaptation
python scripts/train_rc_mamba.py \
  --use_lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1
```

## 📊 Evaluation

### Comprehensive Evaluation Suite

```bash
# Run full evaluation
python scripts/run_experiments.py \
  --experiment comprehensive \
  --model_path checkpoints/rc_mamba_best.pt \
  --output_dir results/

# Specific task evaluation
python scripts/run_experiments.py \
  --experiment needle_in_haystack \
  --sequence_lengths 1000,2000,4000,8000 \
  --retrieval_k 5,10,20
```

### Evaluation Tasks

1. **Long-Context Tasks**
   - Needle-in-haystack retrieval
   - Document summarization
   - Multi-document QA

2. **Multimodal Tasks**
   - Visual question answering
   - Image captioning
   - Audio transcription

3. **Cross-Lingual Tasks**
   - Natural language inference (XNLI)
   - Cross-lingual retrieval
   - Machine translation

4. **Efficiency Analysis**
   - Latency benchmarks
   - Memory usage profiling
   - Compression analysis

### Custom Evaluation

```python
from rc_mamba.eval import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator(
    model=model,
    retriever=retriever,
    device="cuda"
)

# Run specific evaluations
results = evaluator.evaluate_long_context(
    dataset="narrativeqa",
    max_length=4000,
    retrieval_k=10
)

# Generate visualizations
evaluator.plot_results(results, save_dir="plots/")
```

## 🔬 Experiments

### Systematic Experiments

```bash
# Ablation studies
python scripts/run_experiments.py --experiment ablation

# Scaling experiments
python scripts/run_experiments.py --experiment scaling

# Baseline comparisons
python scripts/run_experiments.py --experiment baselines

# Multimodal experiments
python scripts/run_experiments.py --experiment multimodal
```

### Custom Experiments

```python
from scripts.run_experiments import ExperimentRunner

runner = ExperimentRunner()

# Define custom experiment
experiment_config = {
    "name": "custom_retrieval_study",
    "models": ["rc_mamba_base", "rc_mamba_large"],
    "datasets": ["narrativeqa", "hotpotqa"],
    "retrieval_k": [1, 5, 10, 20],
    "metrics": ["accuracy", "f1", "latency"]
}

# Run experiment
results = runner.run_experiment(experiment_config)
```

### Paper Generation

```bash
# Generate research paper with results
python scripts/generate_paper.py \
  --results_dir results/ \
  --output_path paper/rc_mamba_paper.tex \
  --include_appendix \
  --generate_figures
```

## 📚 API Reference

### Core Models

- `RCMambaModel`: Main retrieval-conditioned model
- `MambaSSM`: Core state space model with FiLM conditioning
- `RCMambaConfig`: Model configuration class

### Retrieval Systems

- `MultiModalRetriever`: Cross-modal retrieval system
- `MultiHopRetriever`: Dynamic multi-hop retrieval
- `CrossModalRetriever`: Base cross-modal retrieval

### Training Components

- `PiDPOTrainer`: Advanced preference optimization trainer
- `RCMambaTrainer`: Standard supervised trainer
- `DatasetFactory`: Unified dataset creation interface

### Evaluation Tools

- `ComprehensiveEvaluator`: Multi-dimensional evaluation suite
- `NeedleInHaystackEvaluator`: Long-context evaluation
- `MultimodalEvaluator`: Cross-modal task evaluation

### Utilities

- `DualCodebookQuantizer`: Adaptive quantization system
- `FiLMLayer`: Feature-wise linear modulation
- `AdaptiveBitwidthSelector`: Dynamic compression

## 🔧 Configuration

### Model Configuration

```yaml
# config/model_config.yaml
model:
  name: "rc_mamba_base"
  vocab_size: 32000
  d_model: 768
  n_layers: 12
  
  ssm:
    d_state: 16
    d_conv: 4
    expand_factor: 2
    film_conditioning: true
    
  retrieval:
    embedding_dim: 512
    max_retrieved: 10
    multi_hop: true
    uncertainty_threshold: 0.5
```

### Training Configuration

```yaml
# config/train_config.yaml
training:
  method: "pi_dpo"  # or "standard"
  batch_size: 8
  learning_rate: 1e-4
  num_epochs: 5
  
  pi_dpo:
    sft_weight: 0.7
    dpo_weight: 0.3
    uncertainty_threshold: 0.5
    
  optimization:
    use_lora: true
    lora_rank: 16
    gradient_accumulation_steps: 4
    mixed_precision: true
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black .
isort .
```

### Code Style

- Follow PEP 8 conventions
- Use type hints for all functions
- Add comprehensive docstrings
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use RC-Mamba in your research, please cite:

```bibtex
@inproceedings{rc_mamba_2024,
  title={RC-Mamba: Retrieval-Conditioned State Space Models for Enhanced Reasoning},
  author={Abhay Nori},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```

## 🙏 Acknowledgments

- Mamba team for the original state space model implementation
- Hugging Face for the transformers library
- FAISS team for efficient similarity search
- The open source community for various tools and libraries

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/abhaynori/RC-Mamba/issues)
- **Discussions**: [GitHub Discussions](https://github.com/abhaynori/RC-Mamba/discussions)
- **Email**: anori@uw.edu

---

*RC-Mamba: Bridging the gap between efficient state space models and powerful retrieval mechanisms for next-generation AI systems.*
