"""
DiT (Diffusion Transformer) model implementation for image generation.
Based on the architecture from "Scalable Diffusion Models with Transformers"
Adapted for CIFAR-10 and MNIST datasets.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for transformers."""
    def __init__(self, head_dim, max_seq_len=2048):
        super().__init__()
        # head_dim should be divisible by 2
        assert head_dim % 2 == 0, "head_dim must be divisible by 2"
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # [seq_len, head_dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, head_dim]
        return emb.cos(), emb.sin()


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to query and key.
    
    Args:
        q, k: [B*H, N, D] query and key tensors
        cos, sin: [N, D] rotary embedding (position-wise)
    Returns:
        q_embed, k_embed: [B*H, N, D] rotated query and key
    """
    # cos, sin: [N, D] -> [1, N, D] for broadcasting
    cos = cos[None, :, :]  # [1, N, D]
    sin = sin[None, :, :]  # [1, N, D]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CNNGroupNormEmbedding(nn.Module):
    """CNN + GroupNorm input embedding for CIFAR-10/MNIST."""
    def __init__(self, in_channels, hidden_size, patch_size):
        super().__init__()
        self.patch_size = patch_size
        
        # CNN-based patch embedding
        self.conv = nn.Conv2d(
            in_channels, 
            hidden_size, 
            kernel_size=patch_size, 
            stride=patch_size,
            padding=0
        )
        self.norm = nn.GroupNorm(8, hidden_size)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] image tensor
        Returns:
            [B, N, hidden_size] where N = (H // patch_size) * (W // patch_size)
        """
        # Extract patches using convolution
        patches = self.conv(x)  # [B, hidden_size, H//patch_size, W//patch_size]
        B, C, H_p, W_p = patches.shape
        
        # Flatten spatial dimensions
        patches = patches.view(B, C, H_p * W_p).permute(0, 2, 1)  # [B, N, hidden_size]
        patches = self.norm(patches.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        
        return patches


class AdaLNZero(nn.Module):
    """Adaptive Layer Normalization with zero initialization (for DiT blocks)."""
    def __init__(self, hidden_size, time_emb_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ada_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, hidden_size * 2, bias=True)
        )
        # Initialize zero
        nn.init.zeros_(self.ada_mlp[-1].weight)
        nn.init.zeros_(self.ada_mlp[-1].bias)
        
    def forward(self, x, time_emb):
        """
        Args:
            x: [B, N, hidden_size]
            time_emb: [B, time_emb_dim]
        """
        shift, scale = self.ada_mlp(time_emb).chunk(2, dim=-1)
        x = self.norm(x)
        x = x * (1 + scale[:, None, :]) + shift[:, None, :]
        return x


class DiTBlock(nn.Module):
    """Transformer block with Rotary Embedding and AdaLN."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Separate projection for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, rotary_cos=None, rotary_sin=None):
        """
        Args:
            x: [B, N, hidden_size]
            rotary_cos, rotary_sin: [N] rotary embedding
        """
        # Self-attention with rotary embedding
        residual = x
        x = self.norm1(x)
        
        B, N, C = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, N, D]
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, N, D]
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, N, D]
        
        # Apply rotary embedding
        if rotary_cos is not None and rotary_sin is not None:
            # Reshape for rotary embedding: [B*H, N, D]
            q_flat = q.reshape(B * self.num_heads, N, self.head_dim)
            k_flat = k.reshape(B * self.num_heads, N, self.head_dim)
            q_flat, k_flat = apply_rotary_pos_emb(q_flat, k_flat, rotary_cos, rotary_sin)
            q = q_flat.reshape(B, self.num_heads, N, self.head_dim)
            k = k_flat.reshape(B, self.num_heads, N, self.head_dim)
        
        # Scaled dot-product attention
        scale = (self.head_dim) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # [B, H, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v)  # [B, H, N, D]
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)  # [B, N, H*D]
        out = self.out_proj(out)
        
        x = residual + out
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) model.
    
    Configuration for CIFAR-10 (from Table 4):
    - Patch Size: 2
    - Depth: 10
    - Hidden Size: 256
    - Number of Heads: 8
    - Input Shape: (3, 32, 32)
    - Positional Encoding: Rotary Embedding
    - Input Embedding: CNN + GroupNorm
    """
    def __init__(
        self,
        input_shape=(3, 32, 32),  # (C, H, W)
        patch_size=2,
        depth=10,
        hidden_size=256,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        time_emb_dim=256,
        class_cond=False,
        num_classes=None,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.time_emb_dim = time_emb_dim
        self.class_cond = class_cond
        
        C, H, W = input_shape
        self.num_patches = (H // patch_size) * (W // patch_size)
        
        # Input embedding: CNN + GroupNorm
        self.input_embedding = CNNGroupNormEmbedding(C, hidden_size, patch_size)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Class embedding (optional)
        if class_cond:
            if num_classes is None:
                num_classes = 10  # Default for CIFAR-10
            self.class_embedding = nn.Embedding(num_classes, time_emb_dim)
        else:
            self.class_embedding = None
        
        # Rotary position embedding (per head dimension)
        head_dim = hidden_size // num_heads
        self.rotary_embed = RotaryEmbedding(head_dim, max_seq_len=self.num_patches)
        
        # AdaLN layers
        self.ada_ln_layers = nn.ModuleList([
            AdaLNZero(hidden_size, time_emb_dim) for _ in range(depth)
        ])
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        
        # Final layer norm and output projection
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.output_proj = nn.Linear(hidden_size, C * patch_size * patch_size)
        
    def forward(self, x, t, class_labels=None):
        """
        Args:
            x: [B, C, H, W] noisy image tensor
            t: [B] timestep tensor (continuous time 0 to 1)
            class_labels: [B] optional class labels for conditioning
        Returns:
            [B, C, H, W] predicted velocity field
        """
        B = x.shape[0]
        
        # Convert time to embedding
        # Scale continuous time [0, 1] for sinusoidal embedding
        t_scaled = t * 1000.0  # Scale to reasonable range
        time_emb = self.time_embed(t_scaled)  # [B, time_emb_dim]
        
        # Add class conditioning if needed
        if self.class_cond and class_labels is not None:
            class_emb = self.class_embedding(class_labels)  # [B, time_emb_dim]
            time_emb = time_emb + class_emb
        
        # Input embedding: patches
        x = self.input_embedding(x)  # [B, N, hidden_size]
        
        # Rotary position embedding
        rotary_cos, rotary_sin = self.rotary_embed(x.shape[1], x.device)
        
        # Apply transformer blocks with AdaLN
        for block, ada_ln in zip(self.blocks, self.ada_ln_layers):
            x = ada_ln(x, time_emb)
            x = block(x, rotary_cos, rotary_sin)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Output projection: predict patches
        x = self.output_proj(x)  # [B, N, C * patch_size * patch_size]
        
        # Reshape back to image
        C, H, W = self.input_shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        x = x.reshape(B, patch_h, patch_w, C, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        
        return x


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion models."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        """
        Args:
            time: [B] timestep tensor
        Returns:
            [B, dim] time embedding
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        
        if self.dim % 2 == 1:  # Zero pad if odd dimension
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
            
        return embeddings


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing DiT model...")
    
    # CIFAR-10 configuration
    model_cifar = DiT(
        input_shape=(3, 32, 32),
        patch_size=2,
        depth=10,
        hidden_size=256,
        num_heads=8,
        dropout=0.1,
        class_cond=False,
    )
    
    total_params = count_parameters(model_cifar)
    print(f"CIFAR-10 DiT Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.rand(batch_size)  # Continuous time [0, 1]
    
    with torch.no_grad():
        out = model_cifar(x, t)
    
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    assert out.shape == x.shape, f"Output shape {out.shape} should match input shape {x.shape}"
    
    # MNIST configuration
    model_mnist = DiT(
        input_shape=(1, 28, 28),
        patch_size=2,
        depth=6,
        hidden_size=64,
        num_heads=4,
        dropout=0.1,
        class_cond=False,
    )
    
    total_params_mnist = count_parameters(model_mnist)
    print(f"\nMNIST DiT Parameters: {total_params_mnist:,} ({total_params_mnist / 1e6:.2f}M)")
    
    x_mnist = torch.randn(batch_size, 1, 28, 28)
    t_mnist = torch.rand(batch_size)
    
    with torch.no_grad():
        out_mnist = model_mnist(x_mnist, t_mnist)
    
    print(f"Input shape: {x_mnist.shape}, Output shape: {out_mnist.shape}")
    assert out_mnist.shape == x_mnist.shape, f"Output shape {out_mnist.shape} should match input shape {x_mnist.shape}"
    
    print("\n✓ All tests passed!")
