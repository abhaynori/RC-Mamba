"""
Full Mamba State Space Model Implementation for RC-Mamba.

This module provides a complete implementation of the Mamba SSM architecture
that can be integrated with the FiLM conditioning mechanism in RC-Mamba.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange, repeat


class MambaSSM(nn.Module):
    """Core Mamba State Space Model implementation.
    
    This implements the selective state space mechanism with efficient
    parallel and recurrent modes, designed to be compatible with 
    FiLM conditioning in RC-Mamba.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        
        # Input projections
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        
        # Convolution layer
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # Activation function
        self.activation = "silu"
        self.act = nn.SiLU()
        
        # SSM parameters - these will be modulated by FiLM
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Initialize dt projection
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            
        # A parameter (state evolution matrix)
        A = repeat(
            torch.arange(1, self.d_state + 1),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        film_params: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass of Mamba SSM with optional FiLM conditioning.
        
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, d_model)
            film_params: Optional tuple of (gamma_B, beta_B, gamma_C, beta_C) for FiLM conditioning
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Input projection
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)
        
        # Convolution
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[..., :seq_len]
        x = rearrange(x, "b d l -> b l d")
        x = self.act(x)
        
        # SSM computation
        y = self.ssm(x, film_params=film_params)
        
        # Gating and output projection
        y = y * self.act(z)
        output = self.out_proj(y)
        
        return output
    
    def ssm(
        self, 
        x: torch.Tensor,
        film_params: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Core SSM computation with optional FiLM conditioning."""
        batch_size, seq_len, d_inner = x.shape
        
        # Project input to get dt, B, C
        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # Apply FiLM conditioning if provided
        if film_params is not None:
            gamma_B, beta_B, gamma_C, beta_C = film_params
            # Expand to match sequence length
            gamma_B = gamma_B.unsqueeze(1).expand(-1, seq_len, -1)
            beta_B = beta_B.unsqueeze(1).expand(-1, seq_len, -1)
            gamma_C = gamma_C.unsqueeze(1).expand(-1, seq_len, -1)
            beta_C = beta_C.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Apply FiLM modulation
            B = (1 + gamma_B) * B + beta_B
            C = (1 + gamma_C) * C + beta_C
        
        # Compute dt
        dt = F.softplus(self.dt_proj(dt))
        
        # Get A matrix
        A = -torch.exp(self.A_log.float())
        
        # Discretize A and B
        dA = torch.exp(einsum(dt, A, "b l d, d n -> b l d n"))
        dB = einsum(dt, B, "b l d, b l n -> b l d n")
        
        # Selective scan
        y = selective_scan_fn(x, dA, dB, C, self.D.float())
        
        return y


def einsum(tensor1, tensor2, equation):
    """Helper function for einsum operations."""
    return torch.einsum(equation, tensor1, tensor2)


def selective_scan_fn(
    u: torch.Tensor,
    delta: torch.Tensor, 
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Selective scan implementation.
    
    Args:
        u: Input tensor (batch, seq_len, d_inner)
        delta: Discretized A matrix (batch, seq_len, d_inner, d_state)
        A: A matrix (batch, seq_len, d_inner, d_state) 
        B: B matrix (batch, seq_len, d_inner, d_state)
        C: C matrix (batch, seq_len, d_state)
        D: Skip connection parameter (d_inner,)
    
    Returns:
        Output tensor (batch, seq_len, d_inner)
    """
    batch_size, seq_len, d_inner = u.shape
    d_state = A.shape[-1]
    
    # Initialize state
    x = torch.zeros(batch_size, d_inner, d_state, device=u.device, dtype=u.dtype)
    
    outputs = []
    
    for i in range(seq_len):
        # Update state: x = A * x + B * u
        x = delta[:, i] * x + B[:, i] * u[:, i].unsqueeze(-1)
        
        # Compute output: y = C * x + D * u
        y = torch.sum(C[:, i].unsqueeze(1) * x, dim=-1)
        if D is not None:
            y = y + D * u[:, i]
        
        outputs.append(y)
    
    return torch.stack(outputs, dim=1)


class MambaBlock(nn.Module):
    """Complete Mamba block with normalization and residual connection."""
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.mixer = MambaSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **kwargs
        )
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        film_params: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Forward pass with layer norm and residual connection."""
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.mixer(hidden_states, film_params=film_params)
        return hidden_states + residual
