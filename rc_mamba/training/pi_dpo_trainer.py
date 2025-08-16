"""
Advanced Training Framework for RC-Mamba with π-DPO.

This module implements a comprehensive training pipeline that combines
supervised fine-tuning (SFT) with Direct Preference Optimization (DPO)
using the π-DPO scheduling approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import wandb
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler
from peft import LoraConfig, get_peft_model, TaskType
import warnings


@dataclass
class TrainingConfig:
    """Configuration for RC-Mamba training."""
    # Model parameters
    model_name: str = "rc_mamba"
    d_model: int = 512
    n_layers: int = 8
    d_state: int = 16
    expand: int = 2
    retrieval_dim: int = 256
    vocab_size: int = 50257
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_epochs: int = 10
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # π-DPO parameters
    pi_dpo_enabled: bool = True
    dpo_beta: float = 0.1
    dpo_alpha: float = 2.0
    sft_dpo_ratio: float = 0.5  # Initial ratio of SFT to DPO
    uncertainty_threshold: float = 1.0
    
    # LoRA parameters
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # Retrieval parameters
    max_retrieval_hops: int = 3
    retrieval_update_frequency: int = 100  # steps
    
    # Logging and saving
    output_dir: str = "outputs"
    logging_steps: int = 50
    save_steps: int = 1000
    eval_steps: int = 500
    use_wandb: bool = True
    project_name: str = "rc_mamba_training"
    
    # Data parameters
    max_seq_length: int = 2048
    train_data_path: str = None
    eval_data_path: str = None
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["in_proj", "out_proj", "x_proj", "dt_proj"]


class RCMambaDataset(Dataset):
    """Dataset class for RC-Mamba training."""
    
    def __init__(
        self, 
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        is_preference_data: bool = False
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_preference_data = is_preference_data
        
        # Load data
        if data_path and Path(data_path).exists():
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        else:
            # Create synthetic data for demonstration
            self.data = self._create_synthetic_data()
    
    def _create_synthetic_data(self) -> List[Dict]:
        """Create synthetic training data."""
        synthetic_data = []
        
        # SFT data
        for i in range(1000):
            text = f"This is a synthetic training example {i}. " * 10
            synthetic_data.append({
                "text": text,
                "type": "sft"
            })
        
        # Preference data
        for i in range(500):
            prompt = f"Question: What is the meaning of life example {i}?"
            chosen = f"The meaning of life is a profound philosophical question that has been contemplated for centuries."
            rejected = f"I don't know."
            
            synthetic_data.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "type": "preference"
            })
        
        return synthetic_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if item.get("type") == "preference":
            # Preference data for DPO
            prompt = item["prompt"]
            chosen = item["chosen"]
            rejected = item["rejected"]
            
            # Tokenize
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
            chosen_tokens = self.tokenizer.encode(chosen, add_special_tokens=False)
            rejected_tokens = self.tokenizer.encode(rejected, add_special_tokens=False)
            
            # Combine prompt + response
            chosen_full = prompt_tokens + chosen_tokens
            rejected_full = prompt_tokens + rejected_tokens
            
            # Truncate if necessary
            chosen_full = chosen_full[:self.max_length]
            rejected_full = rejected_full[:self.max_length]
            
            return {
                "input_ids": torch.tensor(prompt_tokens),
                "chosen_ids": torch.tensor(chosen_full),
                "rejected_ids": torch.tensor(rejected_full),
                "type": "preference"
            }
        else:
            # SFT data
            text = item["text"]
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            tokens = tokens[:self.max_length]
            
            return {
                "input_ids": torch.tensor(tokens),
                "labels": torch.tensor(tokens),
                "type": "sft"
            }


def collate_fn(batch):
    """Custom collate function for mixed SFT and preference data."""
    sft_items = [item for item in batch if item["type"] == "sft"]
    pref_items = [item for item in batch if item["type"] == "preference"]
    
    result = {}
    
    if sft_items:
        # Process SFT items
        input_ids = [item["input_ids"] for item in sft_items]
        labels = [item["labels"] for item in sft_items]
        
        # Pad sequences
        max_len = max(len(seq) for seq in input_ids)
        
        padded_input_ids = []
        padded_labels = []
        
        for inp, lab in zip(input_ids, labels):
            pad_len = max_len - len(inp)
            padded_input_ids.append(torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)]))
            padded_labels.append(torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)]))
        
        result["sft"] = {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels)
        }
    
    if pref_items:
        # Process preference items
        input_ids = [item["input_ids"] for item in pref_items]
        chosen_ids = [item["chosen_ids"] for item in pref_items]
        rejected_ids = [item["rejected_ids"] for item in pref_items]
        
        # Pad sequences
        max_len_input = max(len(seq) for seq in input_ids)
        max_len_chosen = max(len(seq) for seq in chosen_ids)
        max_len_rejected = max(len(seq) for seq in rejected_ids)
        
        def pad_sequence(sequences, max_len):
            padded = []
            for seq in sequences:
                pad_len = max_len - len(seq)
                padded.append(torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)]))
            return torch.stack(padded)
        
        result["preference"] = {
            "input_ids": pad_sequence(input_ids, max_len_input),
            "chosen_ids": pad_sequence(chosen_ids, max_len_chosen),
            "rejected_ids": pad_sequence(rejected_ids, max_len_rejected)
        }
    
    return result


class PiDPOTrainer:
    """Advanced trainer implementing π-DPO with retrieval conditioning."""
    
    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        retriever: Any,
        config: TrainingConfig,
        tokenizer: AutoTokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ):
        self.model = model
        self.ref_model = ref_model
        self.retriever = retriever
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.ref_model.to(self.device)
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Setup LoRA if enabled
        if config.use_lora:
            self._setup_lora()
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Setup data loaders
        self._setup_data_loaders()
        
        # Initialize wandb if enabled
        if config.use_wandb:
            self._init_wandb()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')
        
        # π-DPO state
        self.uncertainty_history = []
        self.current_sft_dpo_ratio = config.sft_dpo_ratio
    
    def _setup_lora(self):
        """Setup LoRA adaptation for efficient training."""
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        print(f"LoRA enabled. Trainable parameters: {self.model.print_trainable_parameters()}")
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Optimizer
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Scheduler
        num_training_steps = len(self.train_dataset) // (
            self.config.batch_size * self.config.gradient_accumulation_steps
        ) * self.config.max_epochs
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=self.config.warmup_steps
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps - self.config.warmup_steps,
            eta_min=self.config.learning_rate * 0.1
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_steps]
        )
    
    def _setup_data_loaders(self):
        """Setup training and evaluation data loaders."""
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        if self.eval_dataset:
            self.eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=True
            )
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config.project_name,
            config=self.config.__dict__,
            name=f"{self.config.model_name}_{int(time.time())}"
        )
    
    def compute_sft_loss(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute supervised fine-tuning loss."""
        input_ids = batch_data["input_ids"].to(self.device)
        labels = batch_data["labels"].to(self.device)
        
        # Get retrieval embeddings
        retrieval_queries = []
        for seq in input_ids:
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
            retrieval_queries.append({"text": text})
        
        # Get retrieval embeddings (batch processing)
        if hasattr(self.retriever, 'batch_retrieve'):
            retrieval_embs = self.retriever.batch_retrieve(retrieval_queries)
        else:
            retrieval_embs = []
            for query in retrieval_queries:
                emb = self.retriever(query)
                retrieval_embs.append(emb)
            retrieval_embs = torch.stack(retrieval_embs)
        
        # Forward pass
        logits = self.model(input_ids, retrieval=retrieval_embs)
        
        # Compute loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss
    
    def compute_dpo_loss(self, batch_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """Compute DPO loss with preference data."""
        input_ids = batch_data["input_ids"].to(self.device)
        chosen_ids = batch_data["chosen_ids"].to(self.device)
        rejected_ids = batch_data["rejected_ids"].to(self.device)
        
        batch_size = input_ids.size(0)
        
        # Get retrieval embeddings for prompts
        retrieval_queries = []
        for seq in input_ids:
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
            retrieval_queries.append({"text": text})
        
        if hasattr(self.retriever, 'batch_retrieve'):
            retrieval_embs = self.retriever.batch_retrieve(retrieval_queries)
        else:
            retrieval_embs = []
            for query in retrieval_queries:
                emb = self.retriever(query)
                retrieval_embs.append(emb)
            retrieval_embs = torch.stack(retrieval_embs)
        
        # Forward pass for policy model
        chosen_logits = self.model(chosen_ids, retrieval=retrieval_embs)
        rejected_logits = self.model(rejected_ids, retrieval=retrieval_embs)
        
        # Forward pass for reference model
        with torch.no_grad():
            ref_chosen_logits = self.ref_model(chosen_ids, retrieval=retrieval_embs)
            ref_rejected_logits = self.ref_model(rejected_ids, retrieval=retrieval_embs)
        
        # Compute log probabilities
        def get_log_probs(logits, labels):
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
            return selected_log_probs.sum(dim=-1)
        
        policy_chosen_logps = get_log_probs(chosen_logits[:, :-1], chosen_ids[:, 1:])
        policy_rejected_logps = get_log_probs(rejected_logits[:, :-1], rejected_ids[:, 1:])
        ref_chosen_logps = get_log_probs(ref_chosen_logits[:, :-1], chosen_ids[:, 1:])
        ref_rejected_logps = get_log_probs(ref_rejected_logits[:, :-1], rejected_ids[:, 1:])
        
        # Compute DPO loss
        policy_ratio = policy_chosen_logps - policy_rejected_logps
        ref_ratio = ref_chosen_logps - ref_rejected_logps
        
        logits_diff = self.config.dpo_beta * (policy_ratio - ref_ratio)
        dpo_loss = -F.logsigmoid(logits_diff).mean()
        
        # Compute additional metrics
        with torch.no_grad():
            accuracy = (logits_diff > 0).float().mean()
            chosen_rewards = self.config.dpo_beta * (policy_chosen_logps - ref_chosen_logps)
            rejected_rewards = self.config.dpo_beta * (policy_rejected_logps - ref_rejected_logps)
        
        metrics = {
            "dpo_accuracy": accuracy.item(),
            "chosen_rewards": chosen_rewards.mean().item(),
            "rejected_rewards": rejected_rewards.mean().item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item()
        }
        
        return dpo_loss, metrics
    
    def compute_uncertainty(self, logits: torch.Tensor) -> float:
        """Compute uncertainty measure for π-DPO scheduling."""
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            return entropy.mean().item()
    
    def update_sft_dpo_ratio(self):
        """Update the SFT/DPO ratio based on recent uncertainty."""
        if len(self.uncertainty_history) > 10:
            recent_uncertainty = np.mean(self.uncertainty_history[-10:])
            
            # Higher uncertainty -> more DPO, lower uncertainty -> more SFT
            uncertainty_factor = torch.sigmoid(
                torch.tensor(self.config.dpo_alpha * (recent_uncertainty - self.config.uncertainty_threshold))
            ).item()
            
            # Update ratio (0 = all SFT, 1 = all DPO)
            self.current_sft_dpo_ratio = 1 - uncertainty_factor
    
    def training_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform a single training step with π-DPO."""
        self.model.train()
        total_loss = 0
        metrics = {}
        
        # SFT step
        if "sft" in batch and np.random.random() < self.current_sft_dpo_ratio:
            sft_loss = self.compute_sft_loss(batch["sft"])
            total_loss += sft_loss
            metrics["sft_loss"] = sft_loss.item()
            
            # Compute uncertainty for π-DPO scheduling
            with torch.no_grad():
                logits = self.model(batch["sft"]["input_ids"].to(self.device))
                uncertainty = self.compute_uncertainty(logits)
                self.uncertainty_history.append(uncertainty)
                metrics["uncertainty"] = uncertainty
        
        # DPO step
        if "preference" in batch and np.random.random() >= self.current_sft_dpo_ratio:
            dpo_loss, dpo_metrics = self.compute_dpo_loss(batch["preference"])
            total_loss += dpo_loss
            metrics["dpo_loss"] = dpo_loss.item()
            metrics.update(dpo_metrics)
        
        # Backward pass
        if total_loss > 0:
            total_loss = total_loss / self.config.gradient_accumulation_steps
            total_loss.backward()
            
            metrics["total_loss"] = total_loss.item() * self.config.gradient_accumulation_steps
        
        # Update π-DPO ratio
        if len(self.uncertainty_history) > 0:
            self.update_sft_dpo_ratio()
            metrics["sft_dpo_ratio"] = self.current_sft_dpo_ratio
        
        return metrics
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config.max_epochs} epochs...")
        
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            epoch_metrics = []
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.max_epochs}")
            
            for step, batch in enumerate(progress_bar):
                # Training step
                step_metrics = self.training_step(batch)
                epoch_metrics.append(step_metrics)
                
                # Gradient update
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_metrics = {}
                    for key in step_metrics.keys():
                        values = [m.get(key, 0) for m in epoch_metrics[-self.config.logging_steps:]]
                        avg_metrics[key] = np.mean([v for v in values if v > 0])
                    
                    progress_bar.set_postfix(avg_metrics)
                    
                    if self.config.use_wandb:
                        wandb.log(avg_metrics, step=self.global_step)
                
                # Evaluation
                if self.global_step % self.config.eval_steps == 0 and self.eval_dataset:
                    eval_metrics = self.evaluate()
                    if self.config.use_wandb:
                        wandb.log(eval_metrics, step=self.global_step)
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
        
        print("Training completed!")
        self.save_checkpoint("final")
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the evaluation dataset."""
        if not self.eval_dataset:
            return {}
        
        self.model.eval()
        eval_losses = []
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                if "sft" in batch:
                    loss = self.compute_sft_loss(batch["sft"])
                    eval_losses.append(loss.item())
        
        avg_eval_loss = np.mean(eval_losses) if eval_losses else 0
        
        metrics = {
            "eval_loss": avg_eval_loss,
            "eval_perplexity": np.exp(avg_eval_loss) if avg_eval_loss > 0 else 0
        }
        
        self.model.train()
        return metrics
    
    def save_checkpoint(self, suffix: str = None):
        """Save model checkpoint."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if suffix:
            checkpoint_dir = output_dir / f"checkpoint-{suffix}"
        else:
            checkpoint_dir = output_dir / f"checkpoint-{self.global_step}"
        
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        
        # Save training state
        torch.save({
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "uncertainty_history": self.uncertainty_history,
            "current_sft_dpo_ratio": self.current_sft_dpo_ratio,
        }, checkpoint_dir / "training_state.pt")
        
        # Save config
        with open(checkpoint_dir / "config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        print(f"Checkpoint saved to {checkpoint_dir}")


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer
    from rc_mamba.models.rc_mamba import RCMamba
    from rc_mamba.retrieval.multimodal_retriever import MultiHopRetriever, CrossModalRetriever, MultiModalEncoder
    
    # Initialize components
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    config = TrainingConfig(vocab_size=tokenizer.vocab_size)
    
    # Initialize models
    model = RCMamba(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        d_state=config.d_state,
        expand=config.expand,
        retrieval_dim=config.retrieval_dim
    )
    
    ref_model = RCMamba(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        d_state=config.d_state,
        expand=config.expand,
        retrieval_dim=config.retrieval_dim
    )
    
    # Initialize retriever
    encoder = MultiModalEncoder(embedding_dim=config.retrieval_dim)
    base_retriever = CrossModalRetriever(encoder)
    retriever = MultiHopRetriever(base_retriever)
    
    # Create datasets
    train_dataset = RCMambaDataset(None, tokenizer)
    eval_dataset = RCMambaDataset(None, tokenizer)
    
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
    
    print("Trainer initialized successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
