"""
Simplified UNet for CIFAR10 generation.
Adapted from the original UNetExpert for CIFAR10 (3 channels, 32x32).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """Time step embeddings using sinusoidal encoding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class UNetBlock(nn.Module):
    """Basic UNet block with time embedding."""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)[:, :, None, None]
        
        # First conv
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.relu(h)
        h = h + t_emb  # Add time embedding
        
        # Second conv
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.dropout(h)
        
        # Residual connection
        return F.relu(h + self.res_conv(x))


class UNetExpert(nn.Module):
    """
    Simplified UNet for CIFAR10 (32x32 images, 3 channels).
    Much smaller than Flux/MMDiT - designed for CIFAR10 generation.
    """
    def __init__(
        self,
        in_channels=3,  # CIFAR10 is RGB
        out_channels=3,
        time_emb_dim=128,
        base_channels=32,  # Start with 32 channels (much smaller than original)
        channel_multipliers=[1, 2, 4],  # [32, 64, 128] channels
        num_res_blocks=2,
        dropout=0.1,
        num_timesteps=1000,  # For time scaling
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.channel_multipliers = channel_multipliers
        self.num_res_blocks = num_res_blocks
        self.num_timesteps = num_timesteps
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Input projection
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        ch = base_channels
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(UNetBlock(ch, out_ch, time_emb_dim, dropout))
                ch = out_ch
            if i < len(channel_multipliers) - 1:  # Don't downsample after last block
                self.down_samples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
        
        # Middle block
        self.mid_block1 = UNetBlock(ch, ch, time_emb_dim, dropout)
        self.mid_block2 = UNetBlock(ch, ch, time_emb_dim, dropout)
        
        # Upsampling path - reverse order
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        # Reverse channel multipliers for upsampling (skip the last one)
        up_multipliers = list(reversed(channel_multipliers[:-1]))
        for i, mult in enumerate(up_multipliers):
            out_ch = base_channels * mult
            # First block concatenates skip connection, so input is ch + out_ch
            self.up_blocks.append(UNetBlock(ch + out_ch, out_ch, time_emb_dim, dropout))
            ch = out_ch
            # Additional blocks at same resolution
            for _ in range(num_res_blocks - 1):
                self.up_blocks.append(UNetBlock(ch, ch, time_emb_dim, dropout))
            # Upsample (except for the last level)
            if i < len(up_multipliers) - 1:
                self.up_samples.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
        
        # Final output
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.ReLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )
    
    def forward(self, x, t):
        """
        Args:
            x: [B, C, H, W] noisy image
            t: [B] timestep tensor (continuous time 0 to 1, or integer 0 to num_timesteps-1)
        Returns:
            [B, C, H, W] predicted velocity field
        """
        # Convert continuous time [0, 1] to integer-like timesteps [0, num_timesteps-1]
        # This allows the model to work with both continuous and discrete time
        if t.dtype == torch.float32 or t.dtype == torch.float64:
            # Continuous time: scale to [0, num_timesteps-1]
            t_scaled = t * (self.num_timesteps - 1)
        else:
            t_scaled = t.float()
        
        # Time embedding
        t_emb = self.time_embed(t_scaled)
        
        # Input
        h = self.input_conv(x)
        
        # Downsampling - collect skip connections
        skip_connections = []
        block_idx = 0
        sample_idx = 0
        
        for mult_idx, mult in enumerate(self.channel_multipliers):
            # Process blocks at this level
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[block_idx](h, t_emb)
                block_idx += 1
            # Save skip connection before downsampling (except last level)
            if mult_idx < len(self.channel_multipliers) - 1:
                skip_connections.append(h)
                h = self.down_samples[sample_idx](h)
                sample_idx += 1
        
        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)
        
        # Upsampling - use skip connections
        block_idx = 0
        sample_idx = 0
        
        for mult_idx in range(len(self.channel_multipliers) - 1):
            # First block: concatenate with skip connection
            if skip_connections:
                skip = skip_connections.pop()
                h = F.interpolate(h, size=skip.shape[2:], mode='nearest')
                h = torch.cat([h, skip], dim=1)
            h = self.up_blocks[block_idx](h, t_emb)
            block_idx += 1
            
            # Additional blocks at same resolution
            for _ in range(self.num_res_blocks - 1):
                h = self.up_blocks[block_idx](h, t_emb)
                block_idx += 1
            
            # Upsample (except for the last level)
            if mult_idx < len(self.channel_multipliers) - 2:
                h = self.up_samples[sample_idx](h)
                sample_idx += 1
        
        # Output
        return self.output_conv(h)


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model and count parameters
    model = UNetExpert(in_channels=3, out_channels=3)
    total_params = count_parameters(model)
    print(f"UNet Expert Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.rand(batch_size)  # Continuous time [0, 1]
    out = model(x, t)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")

