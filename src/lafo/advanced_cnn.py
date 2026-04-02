"""
Advanced CNN Variants for LAFO Filtering
Sections 3.2-3.3: ARMABlock, Mamba, RNN variants
"""
import numpy as np
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, GroupNorm, ReLU, SiLU, GELU
import sys
sys.path.insert(0, 'lafo_meanrev/src')


class ARMABlock(nn.Module):
    """
    ARMABlock (Attention-based Recurrent Memory Block)
    Combines convolutional filtering with attention mechanisms for long-range dependencies.
    
    Architecture:
    - Input projection
    - Convolutional filtering
    - Attention mechanism
    - Memory gate
    - Output projection
    """
    
    def __init__(self, channels: int = 64, kernel_size: int = 31,
                 attention_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        
        # Convolutional layer
        self.conv = Conv1d(channels, channels, kernel_size=kernel_size, 
                          padding=(kernel_size-1)//2, groups=channels)  # Depthwise
        
        # Attention mechanism
        self.attention_q = nn.Conv1d(channels, attention_heads * 2, kernel_size=1)
        self.attention_kv = nn.Conv1d(channels, attention_heads * 2, kernel_size=1)
        
        # Memory gate
        self.memory_gate = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            SiLU(),
            nn.Conv1d(channels, channels, kernel_size=1)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.LayerNorm(channels),
            nn.Dropout(dropout)
        )
        
        # Time decay
        self.time_decay = nn.Parameter(torch.ones(1, channels) * 0.95)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, C)
            
        Returns:
            Filtered output (B, T, C)
        """
        B, T, C = x.shape
        
        # Convolutional filtering
        x_conv = self.conv(x)
        
        # Attention mechanism
        Q, K, V = self.attention_q(x), self.attention_kv(x), x
        Q = Q.reshape(B, T, self.attention_heads, -1).permute(0, 2, 3, 1)
        K = K.reshape(B, T, self.attention_heads, -1).permute(0, 2, 3, 1)
        V = V.reshape(B, T, self.attention_heads, -1).permute(0, 2, 3, 1)
        
        # Scaled dot-product attention with causal masking
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(T)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        scores = scores.masked_fill(mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention
        x_attended = torch.matmul(attn, V).permute(0, 3, 1, 2).reshape(B, T, C)
        
        # Memory gate
        x_mem = self.memory_gate(x)
        
        # Combine convolution, attention, and memory
        x_combined = x_conv + x_attended * 0.5 + x_mem * 0.5
        
        # Time decay (exponential forgetting)
        x_combined = x_combined * self.time_decay + x * (1 - self.time_decay)
        
        # Output projection
        output = self.output_proj(x_combined) + x
        
        return output


class DualPathARMABlock(nn.Module):
    """
    Dual-Path ARMA Block for improved filtering.
    Combines low-frequency (long-term) and high-frequency (short-term) paths.
    """
    
    def __init__(self, channels: int = 64, low_path_kernels: List[int] = None):
        super().__init__()
        
        if low_path_kernels is None:
            low_path_kernels = [31, 61, 101]
        
        self.low_path = nn.ModuleList([
            ARMABlock(channels=channels, kernel_size=kernel)
            for kernel in low_path_kernels
        ])
        
        self.high_path = ARMABlock(channels=channels, kernel_size=15)
        
        self.path_weights = nn.Parameter(torch.ones(len(low_path_kernels) + 1) / 
                                          (len(low_path_kernels) + 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, C)
            
        Returns:
            Filtered output (B, T, C)
        """
        B, T, C = x.shape
        
        # Low-frequency paths
        low_outputs = [path(x) for path in self.low_path]
        
        # High-frequency path
        high_output = self.high_path(x)
        
        # Weighted combination
        outputs = [low_outputs[i] * self.path_weights[i] 
                   for i in range(len(low_outputs))]
        outputs.append(high_output * self.path_weights[-1])
        
        return sum(outputs)


class MambaFilter(nn.Module):
    """
    Mamba-style Linear Time-Invariant State Space Model for filtering.
    Uses S6 architecture-inspired selective scanning.
    """
    
    def __init__(self, d_state: int = 16, d_conv: int = 16,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.d_state = d_state
        self.d_conv = d_conv
        
        # Convolutional feature extractor
        self.conv = nn.Conv1d(1, expand * d_state, kernel_size=d_conv, 
                             padding=d_conv-1, groups=expand)
        
        # State space parameters
        self.x = nn.Parameter(torch.randn(d_state))
        self.D = nn.Parameter(torch.randn(d_state))
        self.A = nn.Parameter(torch.randn(d_state))
        
        # Output projection
        self.output_proj = nn.Conv1d(expand * d_state, 1, kernel_size=1)
        
        # Time decay
        self.time_decay = nn.Parameter(torch.ones(d_state) * 0.95)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, 1)
            
        Returns:
            Filtered output (B, T, 1)
        """
        B, T, _ = x.shape
        
        # Convolutional features
        x_conv = self.conv(x.transpose(1, 2))  # (B, C, T)
        x_conv = x_conv.transpose(1, 2)  # (B, T, C)
        
        # Selective scanning
        delta = torch.sigmoid(self.D)
        A = torch.exp(self.A * self.time_decay)
        
        # State space model: x_t = A * x_{t-1} + B * input_t
        x_states = []
        x_states.append(x_conv)
        for _ in range(self.d_state):
            x_prev = x_states[-1] if len(x_states) > 0 else torch.zeros_like(x_conv)
            x_new = A @ x_prev + x_conv
            x_states.append(x_new)
        
        # Combine states
        x_combined = sum(x_states)
        
        # Output projection
        output = self.output_proj(x_combined.transpose(1, 2))
        
        return output


class RNNDualPathCNN(nn.Module):
    """
    Dual-Path CNN with RNN components for temporal modeling.
    """
    
    def __init__(self, channels: int = 64, kernel_size: int = 31,
                 rnn_hidden: int = 64, rnn_layers: int = 1):
        super().__init__()
        
        # CNN path
        self.conv_path = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(channels),
            nn.SiLU()
        )
        
        # RNN path (LSTM or GRU)
        self.rnn = nn.LSTM(channels, rnn_hidden, num_layers=rnn_layers,
                          batch_first=True, dropout=0.1)
        self.rnn_proj = nn.Linear(rnn_hidden, channels)
        
        # Combine paths
        self.combine = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.SiLU()
        )
        
        self.output_proj = nn.Sequential(
            nn.Conv1d(channels, 1, kernel_size=1),
            nn.SiLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, 1)
            
        Returns:
            Filtered output (B, T, 1)
        """
        B, T, _ = x.shape
        
        # CNN path
        x_conv = self.conv_path(x)
        
        # RNN path
        x_rnn = self.rnn(x_conv)[0]  # Get hidden state
        x_rnn = self.rnn_proj(x_rnn)
        
        # Average pooling RNN output back to temporal dimension
        x_rnn = x_rnn.mean(dim=1).unsqueeze(1)  # (B, 1, C)
        
        # Combine
        x_combined = x_conv + x_rnn
        
        # Output projection
        output = self.output_proj(self.combine(x_combined))
        
        return output


class CNNFilterFactory:
    """
    Factory for creating different CNN filter variants.
    """
    
    @staticmethod
    def create_armablock_filter(channels: int = 64, kernel_size: int = 31) -> nn.Module:
        """Create ARMABlock filter."""
        return ARMABlock(channels=channels, kernel_size=kernel_size)
    
    @staticmethod
    def create_dualpath_armablock(channels: int = 64, kernels: List[int] = None) -> nn.Module:
        """Create Dual-Path ARMA block filter."""
        return DualPathARMABlock(channels=channels, low_path_kernels=kernels)
    
    @staticmethod
    def create_mamba_filter(d_state: int = 16, d_conv: int = 16) -> nn.Module:
        """Create Mamba-style filter."""
        return MambaFilter(d_state=d_state, d_conv=d_conv)
    
    @staticmethod
    def create_rnncnn(channels: int = 64, kernel_size: int = 31,
                      rnn_hidden: int = 64, rnn_layers: int = 1) -> nn.Module:
        """Create Dual-Path CNN with RNN."""
        return RNNDualPathCNN(channels=channels, kernel_size=kernel_size,
                              rnn_hidden=rnn_hidden, rnn_layers=rnn_layers)


if __name__ == "__main__":
    # Test the filters
    test_input = torch.randn(1, 1000, 1)
    
    print("Testing ARMABlock...")
    armablock = ARMABlock(channels=64, kernel_size=31)
    output = armablock(test_input)
    print(f"ARMABlock output shape: {output.shape}")
    
    print("\nTesting Dual-Path ARMA...")
    dualpath = DualPathARMABlock(channels=64)
    output = dualpath(test_input)
    print(f"Dual-Path output shape: {output.shape}")
    
    print("\nTesting Mamba Filter...")
    mamba = MambaFilter(d_state=16, d_conv=16)
    output = mamba(test_input)
    print(f"Mamba output shape: {output.shape}")
    
    print("\nTesting RNN-CNN...")
    rnncnn = RNNDualPathCNN(channels=64)
    output = rnncnn(test_input)
    print(f"RNN-CNN output shape: {output.shape}")
    
    print("\nAll tests passed!")
