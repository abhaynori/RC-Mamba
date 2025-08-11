"""
Comprehensive training and evaluation framework for RC-Mamba vs Transformer comparison.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from pathlib import Path
import time
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import defaultdict

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    model_type: str  # "mamba" or "transformer"
    vocab_size: int = 10000
    d_model: int = 512
    n_layers: int = 8
    seq_len: int = 1024
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 10
    train_size: int = 10000
    val_size: int = 2000
    test_size: int = 2000
    seed: int = 42
    retrieval_dim: int = 256
    use_retrieval: bool = True
    save_dir: str = "experiments"
    wandb_project: str = "mamba_vs_transformer"

class SyntheticDataset(Dataset):
    """Synthetic dataset for language modeling experiments."""
    
    def __init__(self, size: int, seq_len: int, vocab_size: int, seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Generate synthetic sequences with some structure
        self.sequences = []
        for _ in range(size):
            # Create sequences with patterns to make learning non-trivial
            seq = torch.randint(0, vocab_size, (seq_len,))
            # Add some patterns: every 10th token copies the 5th previous token
            for i in range(10, seq_len):
                if i % 10 == 0 and i >= 5:
                    seq[i] = seq[i - 5]
            self.sequences.append(seq)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            "input_ids": seq[:-1],
            "labels": seq[1:],
            "retrieval": torch.randn(256)  # Random retrieval embedding
        }

class SimpleTransformer(nn.Module):
    """Simple transformer model for comparison."""
    
    def __init__(self, vocab_size: int, d_model: int = 512, n_layers: int = 8, 
                 n_heads: int = 8, retrieval_dim: int = 256, use_retrieval: bool = True):
        super().__init__()
        self.d_model = d_model
        self.use_retrieval = use_retrieval
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)
        
        # Retrieval conditioning via cross-attention 
        if use_retrieval:
            self.retrieval_proj = nn.Linear(retrieval_dim, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids: torch.Tensor, retrieval: Optional[torch.Tensor] = None):
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings + positional embeddings
        x = self.embed(input_ids) + self.pos_embed[:, :seq_len, :]
        
        # Add retrieval conditioning
        if self.use_retrieval and retrieval is not None:
            retrieval_emb = self.retrieval_proj(retrieval).unsqueeze(1)  # (batch, 1, d_model)
            x = x + retrieval_emb
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits

def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, dataloader, optimizer, device, use_retrieval=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_tokens = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        retrieval = batch["retrieval"].to(device) if use_retrieval else None
        
        optimizer.zero_grad()
        
        logits = model(input_ids, retrieval=retrieval)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * labels.numel()
        total_tokens += labels.numel()
    
    return total_loss / total_tokens

def evaluate(model, dataloader, device, use_retrieval=True):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            retrieval = batch["retrieval"].to(device) if use_retrieval else None
            
            logits = model(input_ids, retrieval=retrieval)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()
    
    return total_loss / total_tokens

def measure_inference_time(model, dataloader, device, num_batches=10):
    """Measure inference time."""
    model.eval()
    times = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            input_ids = batch["input_ids"].to(device)
            retrieval = batch["retrieval"].to(device)
            
            start_time = time.time()
            _ = model(input_ids, retrieval=retrieval)
            torch.cuda.synchronize() if device.type == "cuda" else None
            end_time = time.time()
            
            times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)

def run_experiment(config: ExperimentConfig):
    """Run a single experiment."""
    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset = SyntheticDataset(config.train_size, config.seq_len, config.vocab_size, config.seed)
    val_dataset = SyntheticDataset(config.val_size, config.seq_len, config.vocab_size, config.seed + 1)
    test_dataset = SyntheticDataset(config.test_size, config.seq_len, config.vocab_size, config.seed + 2)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Create model
    if config.model_type == "transformer":
        model = SimpleTransformer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            retrieval_dim=config.retrieval_dim,
            use_retrieval=config.use_retrieval
        )
    elif config.model_type == "mamba":
        # Import enhanced RC-Mamba model
        from enhanced_rc_mamba import EnhancedRCMamba
        model = EnhancedRCMamba(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            retrieval_dim=config.retrieval_dim
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    logger.info(f"Model: {config.model_type}")
    logger.info(f"Parameters: {count_parameters(model):,}")
    
    # Training loop
    results = {
        "config": config.__dict__,
        "train_losses": [],
        "val_losses": [],
        "test_loss": None,
        "inference_time_mean": None,
        "inference_time_std": None,
        "parameter_count": count_parameters(model)
    }
    
    best_val_loss = float("inf")
    
    for epoch in range(config.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, config.use_retrieval)
        
        # Validate
        val_loss = evaluate(model, val_loader, device, config.use_retrieval)
        
        results["train_losses"].append(train_loss)
        results["val_losses"].append(val_loss)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model
            save_path = Path(config.save_dir) / f"{config.model_type}_best.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
    
    # Final evaluation
    test_loss = evaluate(model, test_loader, device, config.use_retrieval)
    results["test_loss"] = test_loss
    
    # Measure inference time
    inf_time_mean, inf_time_std = measure_inference_time(model, test_loader, device)
    results["inference_time_mean"] = inf_time_mean
    results["inference_time_std"] = inf_time_std
    
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Inference Time: {inf_time_mean:.4f}±{inf_time_std:.4f}s")
    
    return results

def plot_training_curves(results_list: List[Dict], save_path: str = "training_curves.png"):
    """Plot training curves for comparison."""
    plt.figure(figsize=(15, 5))
    
    # Training loss
    plt.subplot(1, 3, 1)
    for result in results_list:
        model_type = result["config"]["model_type"]
        plt.plot(result["train_losses"], label=f"{model_type} (train)", alpha=0.7)
        plt.plot(result["val_losses"], label=f"{model_type} (val)", linestyle="--", alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Test loss comparison
    plt.subplot(1, 3, 2)
    model_types = [r["config"]["model_type"] for r in results_list]
    test_losses = [r["test_loss"] for r in results_list]
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_list)))
    
    bars = plt.bar(model_types, test_losses, color=colors, alpha=0.7)
    plt.ylabel("Test Loss")
    plt.title("Final Test Loss")
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, loss in zip(bars, test_losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f"{loss:.3f}", ha="center", va="bottom")
    
    # Inference time comparison
    plt.subplot(1, 3, 3)
    inf_times = [r["inference_time_mean"] for r in results_list]
    inf_stds = [r["inference_time_std"] for r in results_list]
    
    bars = plt.bar(model_types, inf_times, yerr=inf_stds, color=colors, alpha=0.7, capsize=5)
    plt.ylabel("Inference Time (s)")
    plt.title("Inference Time per Batch")
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, time_val in zip(bars, inf_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f"{time_val:.3f}", ha="center", va="bottom")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_parameter_efficiency(results_list: List[Dict], save_path: str = "parameter_efficiency.png"):
    """Plot parameter efficiency analysis."""
    plt.figure(figsize=(12, 4))
    
    # Extract data
    model_types = [r["config"]["model_type"] for r in results_list]
    param_counts = [r["parameter_count"] / 1e6 for r in results_list]  # In millions
    test_losses = [r["test_loss"] for r in results_list]
    inf_times = [r["inference_time_mean"] for r in results_list]
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_list)))
    
    # Parameters vs Test Loss
    plt.subplot(1, 2, 1)
    for i, (model_type, params, loss) in enumerate(zip(model_types, param_counts, test_losses)):
        plt.scatter(params, loss, s=100, color=colors[i], label=model_type, alpha=0.7)
        plt.annotate(model_type, (params, loss), xytext=(5, 5), 
                    textcoords="offset points", fontsize=10)
    
    plt.xlabel("Parameters (M)")
    plt.ylabel("Test Loss")
    plt.title("Parameter Efficiency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Parameters vs Inference Time
    plt.subplot(1, 2, 2)
    for i, (model_type, params, time_val) in enumerate(zip(model_types, param_counts, inf_times)):
        plt.scatter(params, time_val, s=100, color=colors[i], label=model_type, alpha=0.7)
        plt.annotate(model_type, (params, time_val), xytext=(5, 5), 
                    textcoords="offset points", fontsize=10)
    
    plt.xlabel("Parameters (M)")
    plt.ylabel("Inference Time (s)")
    plt.title("Computational Efficiency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def create_results_table(results_list: List[Dict]) -> pd.DataFrame:
    """Create a comprehensive results table."""
    table_data = []
    
    for result in results_list:
        config = result["config"]
        row = {
            "Model": config["model_type"].title(),
            "Parameters (M)": f"{result['parameter_count'] / 1e6:.2f}",
            "Final Train Loss": f"{result['train_losses'][-1]:.4f}",
            "Final Val Loss": f"{result['val_losses'][-1]:.4f}",
            "Test Loss": f"{result['test_loss']:.4f}",
            "Inference Time (s)": f"{result['inference_time_mean']:.4f} ± {result['inference_time_std']:.4f}",
            "Layers": config["n_layers"],
            "Hidden Dim": config["d_model"],
            "Sequence Length": config["seq_len"],
        }
        table_data.append(row)
    
    return pd.DataFrame(table_data)

def plot_loss_landscape(results_list: List[Dict], save_path: str = "loss_landscape.png"):
    """Plot detailed loss landscape analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training dynamics
    ax = axes[0, 0]
    for result in results_list:
        model_type = result["config"]["model_type"]
        epochs = range(1, len(result["train_losses"]) + 1)
        ax.plot(epochs, result["train_losses"], label=f"{model_type} (train)", linewidth=2)
        ax.plot(epochs, result["val_losses"], label=f"{model_type} (val)", 
               linestyle="--", linewidth=2, alpha=0.8)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Dynamics")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    
    # Loss convergence rate
    ax = axes[0, 1]
    for result in results_list:
        model_type = result["config"]["model_type"]
        train_losses = np.array(result["train_losses"])
        # Compute loss reduction rate
        loss_reduction = (train_losses[0] - train_losses) / train_losses[0]
        epochs = range(1, len(loss_reduction) + 1)
        ax.plot(epochs, loss_reduction, label=f"{model_type}", linewidth=2)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Reduction Rate")
    ax.set_title("Convergence Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Performance vs efficiency scatter
    ax = axes[1, 0]
    model_types = [r["config"]["model_type"] for r in results_list]
    test_losses = [r["test_loss"] for r in results_list]
    inf_times = [r["inference_time_mean"] for r in results_list]
    param_counts = [r["parameter_count"] / 1e6 for r in results_list]
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_list)))
    
    for i, (model_type, loss, time_val, params) in enumerate(zip(model_types, test_losses, inf_times, param_counts)):
        ax.scatter(time_val, loss, s=params*20, color=colors[i], alpha=0.7, label=model_type)
        ax.annotate(f"{model_type}\n{params:.1f}M", (time_val, loss), 
                   xytext=(5, 5), textcoords="offset points", fontsize=9)
    
    ax.set_xlabel("Inference Time (s)")
    ax.set_ylabel("Test Loss")
    ax.set_title("Performance vs Efficiency\n(Bubble size = Parameter count)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Overfitting analysis
    ax = axes[1, 1]
    for result in results_list:
        model_type = result["config"]["model_type"]
        train_losses = np.array(result["train_losses"])
        val_losses = np.array(result["val_losses"])
        overfitting = val_losses - train_losses
        epochs = range(1, len(overfitting) + 1)
        ax.plot(epochs, overfitting, label=f"{model_type}", linewidth=2)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation - Training Loss")
    ax.set_title("Overfitting Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    # Run experiments comparing Mamba and Transformer
    
    # Configure experiments
    base_config = ExperimentConfig(
        model_type="transformer",  # Will be overridden
        vocab_size=5000,
        d_model=256,
        n_layers=6,
        seq_len=512,
        batch_size=8,
        learning_rate=1e-4,
        num_epochs=15,
        train_size=5000,
        val_size=1000,
        test_size=1000,
        save_dir="experiments",
    )
    
    # Experiment configurations
    configs = [
        ExperimentConfig(**{**base_config.__dict__, "model_type": "transformer"}),
        ExperimentConfig(**{**base_config.__dict__, "model_type": "mamba"}),
    ]
    
    # Run experiments
    results = []
    for config in configs:
        logger.info(f"Running experiment: {config.model_type}")
        try:
            result = run_experiment(config)
            results.append(result)
            
            # Save individual results
            save_path = Path(config.save_dir) / f"{config.model_type}_results.json"
            with open(save_path, "w") as f:
                json.dump(result, f, indent=2)
                
        except Exception as e:
            logger.error(f"Experiment {config.model_type} failed: {e}")
            continue
    
    if len(results) > 0:
        # Generate plots and analysis
        plot_training_curves(results, "training_curves.png")
        plot_parameter_efficiency(results, "parameter_efficiency.png")
        plot_loss_landscape(results, "loss_landscape.png")
        
        # Create results table
        results_df = create_results_table(results)
        print("\n" + "="*80)
        print("EXPERIMENTAL RESULTS SUMMARY")
        print("="*80)
        print(results_df.to_string(index=False))
        print("="*80)
        
        # Save results table
        results_df.to_csv("results_summary.csv", index=False)
        
        # Save all results
        with open("all_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info("Experiments completed successfully!")
        logger.info("Generated files:")
        logger.info("  - training_curves.png")
        logger.info("  - parameter_efficiency.png") 
        logger.info("  - loss_landscape.png")
        logger.info("  - results_summary.csv")
        logger.info("  - all_results.json")
    else:
        logger.error("No experiments completed successfully!")
