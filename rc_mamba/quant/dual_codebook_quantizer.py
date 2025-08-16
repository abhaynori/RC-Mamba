"""
Dual-Codebook Quantization for RC-Mamba.

This module implements advanced quantization techniques including:
- Uniform Finite Scalar Quantization (FSQ)
- Per-channel k-means FSQ
- Adaptive bit-width selection
- Dynamic quantization during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import math


@dataclass
class QuantizationConfig:
    """Configuration for quantization parameters."""
    uniform_levels: List[int] = None  # [8, 6, 5] for uniform FSQ
    kmeans_clusters: int = 256  # Number of clusters for k-means FSQ
    adaptive_bitwidth: bool = True
    min_bits: int = 2
    max_bits: int = 8
    temperature: float = 1.0
    straight_through: bool = True
    commitment_cost: float = 0.25
    
    def __post_init__(self):
        if self.uniform_levels is None:
            self.uniform_levels = [8, 6, 5]  # Total: 240 levels


class UniformFSQ(nn.Module):
    """Uniform Finite Scalar Quantization."""
    
    def __init__(self, levels: List[int], dim: int = -1, commitment_cost: float = 0.25):
        super().__init__()
        self.levels = levels
        self.dim = dim
        self.commitment_cost = commitment_cost
        
        # Compute total number of codes
        self.num_codes = int(np.prod(levels))
        self.num_dims = len(levels)
        
        # Create codebook bounds
        self.register_buffer('bounds', self._create_bounds())
        
    def _create_bounds(self) -> torch.Tensor:
        """Create quantization bounds for each dimension."""
        bounds = []
        for level in self.levels:
            # Create symmetric bounds around 0
            bound = torch.linspace(-1, 1, level)
            bounds.append(bound)
        return torch.stack(bounds, dim=0)
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize input tensor."""
        original_shape = x.shape
        
        # Ensure we have the right number of dimensions
        if x.size(self.dim) != self.num_dims:
            raise ValueError(f"Expected {self.num_dims} dimensions, got {x.size(self.dim)}")
        
        # Move quantization dimension to last
        if self.dim != -1:
            x = x.transpose(self.dim, -1)
        
        # Quantize each dimension independently
        quantized_dims = []
        indices_dims = []
        
        for i, level in enumerate(self.levels):
            x_dim = x[..., i]
            
            # Clamp to [-1, 1] range
            x_dim = torch.clamp(x_dim, -1, 1)
            
            # Find closest quantization level
            bounds_dim = self.bounds[i]
            distances = torch.abs(x_dim.unsqueeze(-1) - bounds_dim.unsqueeze(0))
            indices = torch.argmin(distances, dim=-1)
            quantized = bounds_dim[indices]
            
            quantized_dims.append(quantized)
            indices_dims.append(indices)
        
        quantized = torch.stack(quantized_dims, dim=-1)
        indices = torch.stack(indices_dims, dim=-1)
        
        # Move dimension back to original position
        if self.dim != -1:
            quantized = quantized.transpose(self.dim, -1)
            indices = indices.transpose(self.dim, -1)
        
        return quantized, indices
    
    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Dequantize from indices."""
        # Move quantization dimension to last
        if self.dim != -1:
            indices = indices.transpose(self.dim, -1)
        
        # Dequantize each dimension
        dequantized_dims = []
        for i, level in enumerate(self.levels):
            indices_dim = indices[..., i]
            bounds_dim = self.bounds[i]
            dequantized_dim = bounds_dim[indices_dim]
            dequantized_dims.append(dequantized_dim)
        
        dequantized = torch.stack(dequantized_dims, dim=-1)
        
        # Move dimension back
        if self.dim != -1:
            dequantized = dequantized.transpose(self.dim, -1)
        
        return dequantized
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with straight-through estimation."""
        quantized, indices = self.quantize(x)
        
        # Straight-through estimator
        quantized_st = x + (quantized - x).detach()
        
        # Commitment loss
        commitment_loss = F.mse_loss(x, quantized.detach()) * self.commitment_cost
        
        return quantized_st, indices, commitment_loss


class KMeansFSQ(nn.Module):
    """K-means based Finite Scalar Quantization with per-channel adaptation."""
    
    def __init__(
        self, 
        num_clusters: int = 256,
        dim: int = -1,
        embedding_dim: int = 512,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # Initialize codebook
        self.register_buffer('codebook', torch.randn(num_clusters, embedding_dim))
        self.register_buffer('cluster_usage', torch.zeros(num_clusters))
        self.register_buffer('cluster_sum', torch.zeros(num_clusters, embedding_dim))
        
        # Per-channel statistics
        self.register_buffer('channel_means', torch.zeros(embedding_dim))
        self.register_buffer('channel_stds', torch.ones(embedding_dim))
        
    def _update_codebook(self, x: torch.Tensor, indices: torch.Tensor):
        """Update codebook using exponential moving averages."""
        if not self.training:
            return
        
        batch_size = x.size(0)
        
        # Flatten for easier processing
        x_flat = x.view(-1, self.embedding_dim)
        indices_flat = indices.view(-1)
        
        # Update cluster usage and sums
        with torch.no_grad():
            # One-hot encoding of indices
            one_hot = F.one_hot(indices_flat, self.num_clusters).float()
            
            # Update usage count
            usage = one_hot.sum(dim=0)
            self.cluster_usage.mul_(self.decay).add_(usage, alpha=1 - self.decay)
            
            # Update cluster sums
            cluster_sum = torch.matmul(one_hot.t(), x_flat)
            self.cluster_sum.mul_(self.decay).add_(cluster_sum, alpha=1 - self.decay)
            
            # Update codebook
            usage_stable = self.cluster_usage + self.epsilon
            self.codebook.copy_(self.cluster_sum / usage_stable.unsqueeze(1))
    
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input per channel."""
        if self.training:
            # Update running statistics
            channel_means = x.mean(dim=tuple(range(x.dim() - 1)), keepdim=True)
            channel_stds = x.std(dim=tuple(range(x.dim() - 1)), keepdim=True) + self.epsilon
            
            self.channel_means.mul_(self.decay).add_(
                channel_means.squeeze(), alpha=1 - self.decay
            )
            self.channel_stds.mul_(self.decay).add_(
                channel_stds.squeeze(), alpha=1 - self.decay
            )
        
        # Normalize
        normalized = (x - self.channel_means) / self.channel_stds
        return normalized
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize input using k-means codebook."""
        # Normalize input
        x_norm = self._normalize_input(x)
        
        # Compute distances to all codebook entries
        distances = torch.cdist(x_norm, self.codebook)
        
        # Find closest codebook entry
        indices = torch.argmin(distances, dim=-1)
        
        # Get quantized values
        quantized = self.codebook[indices]
        
        # Denormalize
        quantized = quantized * self.channel_stds + self.channel_means
        
        return quantized, indices
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with codebook updates."""
        quantized, indices = self.quantize(x)
        
        # Update codebook
        self._update_codebook(x, indices)
        
        # Straight-through estimator
        quantized_st = x + (quantized - x).detach()
        
        # Commitment loss
        commitment_loss = F.mse_loss(x, quantized.detach()) * self.commitment_cost
        
        return quantized_st, indices, commitment_loss


class AdaptiveBitwidthSelector(nn.Module):
    """Adaptive bit-width selection based on input complexity."""
    
    def __init__(
        self,
        input_dim: int,
        min_bits: int = 2,
        max_bits: int = 8,
        temperature: float = 1.0
    ):
        super().__init__()
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.temperature = temperature
        
        # Network to predict optimal bit-width
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, max_bits - min_bits + 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Predict optimal bit-width for input."""
        # Compute input complexity measures
        input_stats = torch.cat([
            x.mean(dim=1, keepdim=True),
            x.std(dim=1, keepdim=True),
            x.min(dim=1, keepdim=True)[0],
            x.max(dim=1, keepdim=True)[0]
        ], dim=1)
        
        # Predict bit-width logits
        logits = self.predictor(input_stats)
        
        # Sample bit-width
        if self.training:
            # Gumbel softmax for differentiable sampling
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            soft_selection = F.softmax((logits + gumbel_noise) / self.temperature, dim=-1)
            
            # Straight-through to get hard selection
            hard_selection = F.one_hot(soft_selection.argmax(dim=-1), logits.size(-1)).float()
            selection = soft_selection + (hard_selection - soft_selection).detach()
            
            bits = torch.sum(selection * torch.arange(
                self.min_bits, self.max_bits + 1, device=x.device
            ).float(), dim=-1)
            
            return bits, logits
        else:
            # Deterministic selection during inference
            bits = F.softmax(logits, dim=-1).argmax(dim=-1) + self.min_bits
            return bits, logits


class DualCodebookQuantizer(nn.Module):
    """Dual-codebook quantization combining uniform FSQ and k-means FSQ."""
    
    def __init__(self, config: QuantizationConfig, embedding_dim: int):
        super().__init__()
        self.config = config
        self.embedding_dim = embedding_dim
        
        # Initialize quantizers
        self.uniform_fsq = UniformFSQ(
            levels=config.uniform_levels,
            commitment_cost=config.commitment_cost
        )
        
        self.kmeans_fsq = KMeansFSQ(
            num_clusters=config.kmeans_clusters,
            embedding_dim=embedding_dim,
            commitment_cost=config.commitment_cost
        )
        
        # Adaptive bit-width selector
        if config.adaptive_bitwidth:
            self.bitwidth_selector = AdaptiveBitwidthSelector(
                input_dim=embedding_dim,
                min_bits=config.min_bits,
                max_bits=config.max_bits,
                temperature=config.temperature
            )
        else:
            self.bitwidth_selector = None
        
        # Codebook selection network
        self.codebook_selector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 2)  # Binary choice: uniform vs k-means
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with dual-codebook quantization."""
        batch_size, seq_len, embed_dim = x.shape
        
        # Predict codebook selection
        codebook_logits = self.codebook_selector(x.mean(dim=1))  # Pool over sequence
        
        if self.training:
            # Gumbel softmax for differentiable codebook selection
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(codebook_logits) + 1e-8) + 1e-8)
            soft_selection = F.softmax((codebook_logits + gumbel_noise) / self.config.temperature, dim=-1)
            
            # Straight-through
            hard_selection = F.one_hot(soft_selection.argmax(dim=-1), 2).float()
            codebook_weights = soft_selection + (hard_selection - soft_selection).detach()
        else:
            codebook_weights = F.softmax(codebook_logits, dim=-1)
        
        # Apply both quantizers
        uniform_out, uniform_indices, uniform_loss = self.uniform_fsq(x)
        kmeans_out, kmeans_indices, kmeans_loss = self.kmeans_fsq(x)
        
        # Weighted combination
        uniform_weight = codebook_weights[:, 0].unsqueeze(1).unsqueeze(2)
        kmeans_weight = codebook_weights[:, 1].unsqueeze(1).unsqueeze(2)
        
        quantized = uniform_weight * uniform_out + kmeans_weight * kmeans_out
        commitment_loss = uniform_weight.squeeze() * uniform_loss + kmeans_weight.squeeze() * kmeans_loss
        
        # Adaptive bit-width
        if self.bitwidth_selector:
            target_bits, bitwidth_logits = self.bitwidth_selector(x.view(-1, embed_dim))
            bitwidth_loss = F.cross_entropy(
                bitwidth_logits, 
                torch.randint(0, self.config.max_bits - self.config.min_bits + 1, 
                             (bitwidth_logits.size(0),), device=x.device)
            )
        else:
            target_bits = torch.tensor(4.0)  # Default 4 bits
            bitwidth_loss = torch.tensor(0.0)
        
        return {
            "quantized": quantized,
            "uniform_indices": uniform_indices,
            "kmeans_indices": kmeans_indices,
            "codebook_weights": codebook_weights,
            "target_bits": target_bits,
            "commitment_loss": commitment_loss.mean(),
            "bitwidth_loss": bitwidth_loss,
            "total_loss": commitment_loss.mean() + bitwidth_loss
        }
    
    def get_compression_ratio(self) -> float:
        """Estimate compression ratio achieved."""
        # Uniform FSQ compression
        uniform_bits = math.log2(self.uniform_fsq.num_codes)
        
        # K-means FSQ compression  
        kmeans_bits = math.log2(self.kmeans_fsq.num_clusters)
        
        # Original representation (assuming 32-bit float)
        original_bits = 32
        
        # Weighted average compression ratio
        avg_bits = (uniform_bits + kmeans_bits) / 2
        compression_ratio = original_bits / avg_bits
        
        return compression_ratio
    
    def analyze_usage(self) -> Dict[str, float]:
        """Analyze codebook usage patterns."""
        with torch.no_grad():
            # K-means cluster usage
            kmeans_usage = self.kmeans_fsq.cluster_usage
            kmeans_entropy = -torch.sum(
                F.softmax(kmeans_usage, dim=0) * F.log_softmax(kmeans_usage, dim=0)
            ).item()
            
            # Usage statistics
            stats = {
                "kmeans_entropy": kmeans_entropy,
                "kmeans_active_clusters": (kmeans_usage > 0).sum().item(),
                "kmeans_max_usage": kmeans_usage.max().item(),
                "kmeans_min_usage": kmeans_usage.min().item(),
                "compression_ratio": self.get_compression_ratio()
            }
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    # Test dual-codebook quantizer
    config = QuantizationConfig(
        uniform_levels=[8, 6, 5],
        kmeans_clusters=256,
        adaptive_bitwidth=True
    )
    
    quantizer = DualCodebookQuantizer(config, embedding_dim=512)
    
    # Test input
    x = torch.randn(4, 100, 512)  # (batch, seq_len, embed_dim)
    
    # Forward pass
    output = quantizer(x)
    
    print("Quantization Results:")
    print(f"Input shape: {x.shape}")
    print(f"Quantized shape: {output['quantized'].shape}")
    print(f"Commitment loss: {output['commitment_loss'].item():.4f}")
    print(f"Bitwidth loss: {output['bitwidth_loss'].item():.4f}")
    print(f"Total loss: {output['total_loss'].item():.4f}")
    print(f"Codebook weights: {output['codebook_weights']}")
    
    # Usage analysis
    stats = quantizer.analyze_usage()
    print(f"\nUsage Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nEstimated compression ratio: {quantizer.get_compression_ratio():.2f}x")
