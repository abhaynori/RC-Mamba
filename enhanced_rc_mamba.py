"""
Enhanced RC-Mamba implementation with working SSM layers.
This implementation provides a functional Mamba-like architecture for comparison.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SimpleSSM(nn.Module):
    """Simplified State Space Model implementation."""
    
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        
        # Linear projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=3, padding=1, groups=self.d_inner)
        
        # SSM parameters
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Projection matrices that will be modulated by FiLM
        self.B_proj = nn.Linear(self.d_inner, d_state)
        self.C_proj = nn.Linear(self.d_inner, d_state)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        # Initialize A matrix to be stable
        with torch.no_grad():
            A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
            self.A_log.copy_(torch.log(A))
    
    def forward(self, x: torch.Tensor, B_mod: Optional[torch.Tensor] = None, 
                C_mod: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            B_mod: (batch, d_state) - B matrix modulation
            C_mod: (batch, d_state) - C matrix modulation
        """
        batch, seq_len, _ = x.shape
        
        # Input projection
        x_and_res = self.in_proj(x)  # (batch, seq_len, 2 * d_inner)
        x_ssm, x_res = x_and_res.split(self.d_inner, dim=-1)
        
        # Apply activation and convolution
        x_ssm = F.silu(x_ssm)
        x_ssm = x_ssm.transpose(1, 2)  # (batch, d_inner, seq_len)
        x_ssm = self.conv1d(x_ssm)
        x_ssm = x_ssm.transpose(1, 2)  # (batch, seq_len, d_inner)
        
        # SSM computation with optional FiLM modulation
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Compute B and C matrices
        B = self.B_proj(x_ssm)  # (batch, seq_len, d_state)
        C = self.C_proj(x_ssm)  # (batch, seq_len, d_state)
        
        # Apply FiLM modulation if provided
        if B_mod is not None:
            B = B + B_mod.unsqueeze(1)  # Broadcast over sequence length
        if C_mod is not None:
            C = C + C_mod.unsqueeze(1)  # Broadcast over sequence length
        
        # Simplified SSM computation (this is a approximation for demonstration)
        # In practice, you would use specialized CUDA kernels for efficiency
        y = self._ssm_scan(x_ssm, A, B, C)
        
        # Skip connection and output projection
        y = y * F.silu(x_res)
        y = self.out_proj(y)
        
        return y
    
    def _ssm_scan(self, x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """Simplified SSM scan operation."""
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # Initialize state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        dt = 0.1  # Fixed discretization step
        
        for t in range(seq_len):
            # Discretize A matrix
            A_bar = torch.exp(A * dt)  # (d_inner, d_state)
            
            # Update state: h = A_bar * h + B * x
            x_t = x[:, t:t+1, :].transpose(1, 2)  # (batch, d_inner, 1)
            B_t = B[:, t, :].unsqueeze(1)  # (batch, 1, d_state)
            
            h = A_bar.unsqueeze(0) * h + x_t @ B_t  # (batch, d_inner, d_state)
            
            # Compute output: y = C * h
            C_t = C[:, t, :].unsqueeze(-1)  # (batch, d_state, 1)
            y_t = (h @ C_t).squeeze(-1)  # (batch, d_inner)
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)
        
        # Add skip connection
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x
        
        return y


class EnhancedFiLMModulator(nn.Module):
    """Enhanced FiLM modulator with better conditioning."""
    
    def __init__(self, retrieval_dim: int, d_state: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(d_state * 2, 128)
        
        self.layers = nn.Sequential(
            nn.Linear(retrieval_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2 * d_state)  # For B and C modulation
        )
        
        # Initialize to small values
        with torch.no_grad():
            self.layers[-1].weight.mul_(0.1)
            self.layers[-1].bias.zero_()
    
    def forward(self, retrieval: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            retrieval: (batch, retrieval_dim)
        Returns:
            B_mod, C_mod: each (batch, d_state)
        """
        modulation = self.layers(retrieval)
        B_mod, C_mod = modulation.chunk(2, dim=-1)
        return B_mod, C_mod


class EnhancedRCMambaBlock(nn.Module):
    """Enhanced RC-Mamba block with working SSM."""
    
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, retrieval_dim: int = 256):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # SSM layer
        self.ssm = SimpleSSM(d_model, d_state, expand)
        
        # FiLM modulator
        self.modulator = EnhancedFiLMModulator(retrieval_dim, d_state)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, retrieval: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            retrieval: (batch, retrieval_dim)
        """
        # Get FiLM parameters
        B_mod, C_mod = self.modulator(retrieval)
        
        # Apply layer norm
        x_norm = self.norm(x)
        
        # SSM with FiLM conditioning
        y = self.ssm(x_norm, B_mod, C_mod)
        
        # Residual connection
        return x + y


class EnhancedRCMamba(nn.Module):
    """Enhanced RC-Mamba model with working implementation."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 8,
        d_state: int = 16,
        expand: int = 2,
        retrieval_dim: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.retrieval_dim = retrieval_dim
        
        # Token embedding
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # Mamba blocks
        self.layers = nn.ModuleList([
            EnhancedRCMambaBlock(d_model, d_state, expand, retrieval_dim)
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embed.weight
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        # Initialize embeddings
        nn.init.normal_(self.embed.weight, std=0.02)
        
        # Initialize output projection
        nn.init.normal_(self.lm_head.weight, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, retrieval: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
            retrieval: (batch, retrieval_dim)
        """
        batch_size, seq_len = input_ids.shape
        
        # Default retrieval if not provided
        if retrieval is None:
            retrieval = torch.zeros(batch_size, self.retrieval_dim, 
                                   device=input_ids.device, dtype=self.embed.weight.dtype)
        
        # Token embeddings
        x = self.embed(input_ids)  # (batch, seq_len, d_model)
        
        # Pass through Mamba blocks
        for layer in self.layers:
            x = layer(x, retrieval)
        
        # Final layer norm and projection
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        retrieval: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressive generation."""
        self.eval()
        
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Get logits for the last token
            logits = self.forward(generated, retrieval)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        
        return generated
