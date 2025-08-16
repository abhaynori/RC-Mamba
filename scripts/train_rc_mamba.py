"""
Main Training Script for RC-Mamba Research Project.

This script orchestrates the complete training pipeline including:
- Model initialization with FiLM conditioning
- Multimodal retrieval setup
- π-DPO training with LoRA adaptation
- Comprehensive evaluation across all tasks
- Experiment tracking and visualization
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import wandb
from typing import Dict, Any
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rc_mamba.models.rc_mamba import RCMamba
from rc_mamba.retrieval.multimodal_retriever import (
    MultiModalEncoder, CrossModalRetriever, MultiHopRetriever
)
from rc_mamba.training.pi_dpo_trainer import PiDPOTrainer, TrainingConfig, RCMambaDataset
from rc_mamba.eval.comprehensive_evaluator import ComprehensiveEvaluator
from rc_mamba.quant.dual_codebook_quantizer import DualCodebookQuantizer, QuantizationConfig
from rc_mamba.data.datasets import DatasetFactory, create_research_dataloaders


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RC-Mamba Training and Evaluation")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="rc_mamba_research", 
                       help="Model name for saving")
    parser.add_argument("--d_model", type=int, default=512, 
                       help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=8, 
                       help="Number of layers")
    parser.add_argument("--d_state", type=int, default=16, 
                       help="State dimension")
    parser.add_argument("--expand", type=int, default=2, 
                       help="Expansion factor")
    parser.add_argument("--retrieval_dim", type=int, default=256, 
                       help="Retrieval embedding dimension")
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, 
                       help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=5, 
                       help="Maximum training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, 
                       help="Gradient accumulation steps")
    parser.add_argument("--max_seq_length", type=int, default=2048, 
                       help="Maximum sequence length")
    
    # π-DPO arguments
    parser.add_argument("--pi_dpo_enabled", action="store_true", 
                       help="Enable π-DPO training")
    parser.add_argument("--dpo_beta", type=float, default=0.1, 
                       help="DPO beta parameter")
    parser.add_argument("--dpo_alpha", type=float, default=2.0, 
                       help="DPO alpha parameter")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", default=True,
                       help="Use LoRA adaptation")
    parser.add_argument("--lora_r", type=int, default=16, 
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, 
                       help="LoRA alpha")
    
    # Quantization arguments
    parser.add_argument("--use_quantization", action="store_true", 
                       help="Enable dual-codebook quantization")
    parser.add_argument("--adaptive_bitwidth", action="store_true", 
                       help="Enable adaptive bit-width selection")
    
    # Retrieval arguments
    parser.add_argument("--max_retrieval_hops", type=int, default=3, 
                       help="Maximum retrieval hops")
    parser.add_argument("--retrieval_corpus_size", type=int, default=10000, 
                       help="Size of retrieval corpus")
    
    # Evaluation arguments
    parser.add_argument("--eval_only", action="store_true", 
                       help="Only run evaluation")
    parser.add_argument("--eval_checkpoint", type=str, 
                       help="Checkpoint path for evaluation")
    parser.add_argument("--eval_datasets", nargs="+", 
                       default=["needle", "multimodal", "crosslingual", "efficiency"],
                       help="Datasets to evaluate on")
    
    # I/O arguments
    parser.add_argument("--output_dir", type=str, default="outputs", 
                       help="Output directory")
    parser.add_argument("--data_dir", type=str, default="data", 
                       help="Data directory")
    parser.add_argument("--use_wandb", action="store_true", default=True,
                       help="Use Weights & Biases logging")
    parser.add_argument("--project_name", type=str, default="rc_mamba_neurips", 
                       help="W&B project name")
    
    # Compute arguments
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device to use (auto/cuda/cpu)")
    parser.add_argument("--mixed_precision", action="store_true", 
                       help="Use mixed precision training")
    parser.add_argument("--compile_model", action="store_true", 
                       help="Compile model with torch.compile")
    
    return parser.parse_args()


def setup_device(device_arg: str):
    """Setup compute device."""
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def initialize_tokenizer():
    """Initialize tokenizer."""
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def initialize_models(args, tokenizer, device):
    """Initialize RC-Mamba model and reference model."""
    print("Initializing RC-Mamba models...")
    
    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "d_state": args.d_state,
        "expand": args.expand,
        "retrieval_dim": args.retrieval_dim,
    }
    
    # Main model
    model = RCMamba(**model_config)
    model.to(device)
    
    # Reference model (copy of main model)
    ref_model = RCMamba(**model_config)
    ref_model.load_state_dict(model.state_dict())
    ref_model.to(device)
    
    # Compile models if requested
    if args.compile_model:
        print("Compiling models...")
        model = torch.compile(model)
        ref_model = torch.compile(ref_model)
    
    # Add quantization if enabled
    if args.use_quantization:
        print("Setting up quantization...")
        quant_config = QuantizationConfig(
            adaptive_bitwidth=args.adaptive_bitwidth
        )
        quantizer = DualCodebookQuantizer(quant_config, args.retrieval_dim)
        model.quantizer = quantizer
        print(f"Quantization enabled. Compression ratio: {quantizer.get_compression_ratio():.2f}x")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {num_params:,} parameters")
    
    return model, ref_model


def initialize_retrieval_system(args, device):
    """Initialize multimodal retrieval system."""
    print("Initializing retrieval system...")
    
    # Create multimodal encoder
    encoder = MultiModalEncoder(
        embedding_dim=args.retrieval_dim,
        device=device
    )
    
    # Create base retriever
    base_retriever = CrossModalRetriever(encoder)
    
    # Create multi-hop retriever
    retriever = MultiHopRetriever(
        base_retriever,
        max_hops=args.max_retrieval_hops
    )
    
    # Load corpus data for retrieval
    print("Loading retrieval corpus...")
    try:
        corpus_data = DatasetFactory.create_multimodal_dataset(
            max_samples_per_modality=args.retrieval_corpus_size // 3
        )
        
        # Add corpus to retriever
        corpus_items = []
        for i in range(min(len(corpus_data), args.retrieval_corpus_size)):
            item = corpus_data[i]
            corpus_items.append(item)
        
        # Convert to format expected by retriever
        retrieval_items = []
        for item in corpus_items[:100]:  # Limit for demo
            if item.get("modality") == "text":
                retrieval_items.append({"text": item.get("premise", "") + " " + item.get("hypothesis", "")})
            elif item.get("modality") == "vision":
                retrieval_items.append({"text": item.get("question", ""), "image": item.get("image")})
        
        base_retriever.add_corpus(retrieval_items)
        print(f"Added {len(retrieval_items)} items to retrieval corpus")
        
    except Exception as e:
        warnings.warn(f"Could not load retrieval corpus: {e}")
    
    return retriever


def create_training_config(args) -> TrainingConfig:
    """Create training configuration."""
    return TrainingConfig(
        model_name=args.model_name,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        expand=args.expand,
        retrieval_dim=args.retrieval_dim,
        vocab_size=0,  # Will be set based on tokenizer
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_epochs=args.max_epochs,
        max_seq_length=args.max_seq_length,
        pi_dpo_enabled=args.pi_dpo_enabled,
        dpo_beta=args.dpo_beta,
        dpo_alpha=args.dpo_alpha,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_retrieval_hops=args.max_retrieval_hops,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        project_name=args.project_name,
    )


def run_training(args, model, ref_model, retriever, tokenizer, device):
    """Run the training process."""
    print("Starting training...")
    
    # Create training configuration
    config = create_training_config(args)
    config.vocab_size = tokenizer.vocab_size
    
    # Create datasets
    print("Loading training datasets...")
    train_dataset = RCMambaDataset(
        data_path=None,  # Use synthetic data
        tokenizer=tokenizer,
        max_length=args.max_seq_length
    )
    
    eval_dataset = RCMambaDataset(
        data_path=None,  # Use synthetic data
        tokenizer=tokenizer,
        max_length=args.max_seq_length
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # Initialize trainer
    trainer = PiDPOTrainer(
        model=model,
        ref_model=ref_model,
        retriever=retriever,
        config=config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    # Run training
    trainer.train()
    
    print("Training completed!")
    return trainer


def run_evaluation(args, model, retriever, tokenizer, device):
    """Run comprehensive evaluation."""
    print("Starting evaluation...")
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(
        tokenizer=tokenizer,
        output_dir=os.path.join(args.output_dir, "evaluation_results")
    )
    
    # Run evaluation on selected datasets
    if "all" in args.eval_datasets:
        results = evaluator.run_full_evaluation(model, retriever)
    else:
        results = {}
        
        if "needle" in args.eval_datasets:
            print("Running needle-in-haystack evaluation...")
            results["needle"] = evaluator.needle_evaluator.evaluate_model(model, retriever)
        
        if "multimodal" in args.eval_datasets:
            print("Running multimodal evaluation...")
            results["multimodal"] = evaluator.multimodal_evaluator.evaluate_vqa(model, retriever)
        
        if "crosslingual" in args.eval_datasets:
            print("Running cross-lingual evaluation...")
            results["crosslingual"] = evaluator.crosslingual_evaluator.evaluate_xnli(model, retriever)
        
        if "efficiency" in args.eval_datasets:
            print("Running efficiency evaluation...")
            results["efficiency"] = (
                evaluator.efficiency_evaluator.measure_latency(model) +
                evaluator.efficiency_evaluator.measure_memory_usage(model)
            )
    
    # Save and visualize results
    evaluator.save_results(results)
    evaluator.create_visualizations(results)
    
    # Print summary
    print("\nEvaluation Results Summary:")
    print("=" * 50)
    
    for task, task_results in results.items():
        print(f"\n{task.upper()}:")
        for result in task_results:
            print(f"  {result.metric_name}: {result.score:.4f}")
    
    return results


def save_experiment_config(args, output_dir):
    """Save experiment configuration."""
    config_path = Path(output_dir) / "experiment_config.json"
    
    config_dict = vars(args).copy()
    # Convert any non-serializable objects
    for key, value in config_dict.items():
        if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
            config_dict[key] = str(value)
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Experiment configuration saved to {config_path}")


def main():
    """Main function."""
    print("="*70)
    print("RC-Mamba: Retrieval-Conditioned Mamba for NeurIPS Research")
    print("="*70)
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save experiment configuration
    save_experiment_config(args, output_dir)
    
    # Setup device
    device = setup_device(args.device)
    
    # Initialize tokenizer
    tokenizer = initialize_tokenizer()
    
    # Initialize models
    model, ref_model = initialize_models(args, tokenizer, device)
    
    # Initialize retrieval system
    retriever = initialize_retrieval_system(args, device)
    
    # Load checkpoint if provided for evaluation
    if args.eval_checkpoint:
        print(f"Loading checkpoint from {args.eval_checkpoint}...")
        checkpoint = torch.load(args.eval_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Checkpoint loaded successfully!")
    
    try:
        if args.eval_only:
            # Run evaluation only
            results = run_evaluation(args, model, retriever, tokenizer, device)
        else:
            # Run training
            trainer = run_training(args, model, ref_model, retriever, tokenizer, device)
            
            # Run evaluation after training
            print("\nRunning post-training evaluation...")
            results = run_evaluation(args, model, retriever, tokenizer, device)
        
        print("\nExperiment completed successfully!")
        
        # Log final results to wandb if enabled
        if args.use_wandb:
            wandb.finish()
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        if args.use_wandb:
            wandb.finish()
    except Exception as e:
        print(f"\nExperiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        if args.use_wandb:
            wandb.finish()
        sys.exit(1)


if __name__ == "__main__":
    main()
