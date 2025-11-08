from __future__ import annotations

import torch
import torch.nn as nn


class GaussianFourierFeatures(nn.Module):
    """Gaussian random Fourier feature mapping for 2D inputs.

    Transforms coordinates x in R^2 into a higher-dimensional embedding:
        [sin(2π xB), cos(2π xB)]
    where B ~ N(0, sigma^2).

    Args:
        in_dim: Input dimension (default 2 for complex plane coordinates).
        num_feats: Number of Fourier frequencies (embedding has size 2*num_feats).
        sigma: Standard deviation of the Gaussian used to sample B.
    """
    def __init__(self, in_dim=2, num_feats=256, sigma=5.0):
        super().__init__()
        B = torch.randn(in_dim, num_feats) * sigma
        self.register_buffer("B", B)

    def forward(self, x):
        proj = (2 * torch.pi) * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class MultiScaleGaussianFourierFeatures(nn.Module):
    """Multi-scale Gaussian Fourier feature mapping.

    Like `GaussianFourierFeatures`, but concatenates multiple Gaussian matrices
    with different sigmas to provide a spectrum of frequencies.

    Args:
        in_dim: Input dimension.
        num_feats: Total number of frequencies across all scales.
        sigmas: Tuple of sigma values (one per scale).
        seed: Seed used for deterministic sampling of B.
    """
    def __init__(self, in_dim=2, num_feats=512, sigmas=(2.0, 6.0, 10.0), seed=0):
        super().__init__()
        k = len(sigmas)
        per = [num_feats // k] * k
        per[0] += num_feats - sum(per)

        Bs = []
        g = torch.Generator()
        g.manual_seed(seed)
        for s, m in zip(sigmas, per):
            B = torch.randn(in_dim, m, generator=g) * s
            Bs.append(B)

        self.register_buffer("B", torch.cat(Bs, dim=1))

    def forward(self, x):
        proj = (2 * torch.pi) * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block with optional dropout.

    Uses LayerNorm + Linear + activation + (optional dropout) + LayerNorm + Linear,
    and adds the result back to the input.

    The second linear layer is initialized to zero to start close to identity,
    which usually stabilizes deep residual MLP training.

    Args:
        dim: Hidden dimension.
        act: Activation name ("relu" or "silu").
        dropout: Dropout probability (0.0 disables dropout).
    """
    def __init__(self, dim: int, act: str = "silu", dropout: float = 0.0):
        super().__init__()
        activation = nn.ReLU if act.lower() == "relu" else nn.SiLU

        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)

        self.ln2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)

        self.act = activation()
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h = self.ln1(x)
        h = self.act(self.fc1(h))
        h = self.drop(h)
        h = self.ln2(h)
        h = self.fc2(h)
        return x + h


class MLPRes(nn.Module):
    """Residual MLP for 2D coordinate regression (no Fourier features).

    Args:
        hidden_dim: Hidden dimension.
        num_blocks: Number of residual blocks.
        act: Activation name ("relu" or "silu").
        dropout: Dropout probability.
        out_dim: Output dimension (typically 1).
    """
    def __init__(self, hidden_dim=256, num_blocks=8, act="silu", dropout=0.0, out_dim=1):
        super().__init__()
        activation = nn.ReLU if act.lower() == "relu" else nn.SiLU

        self.in_proj = nn.Linear(2, hidden_dim)
        self.in_act = activation()

        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, act=act, dropout=dropout)
            for _ in range(num_blocks)
        ])

        self.out_ln = nn.LayerNorm(hidden_dim)
        self.out_act = activation()
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.in_act(self.in_proj(x))
        x = self.blocks(x)
        x = self.out_act(self.out_ln(x))
        return self.out_proj(x)


class MLPFourierRes(nn.Module):
    """Residual MLP with multi-scale Fourier features for coordinate regression.
    
    Args:
        num_feats: Number of Fourier frequencies.
        sigma: Largest sigma used in multi-scale embedding (other scales are fixed).
        hidden_dim: Hidden dimension.
        num_blocks: Number of residual blocks.
        act: Activation name ("relu" or "silu").
        dropout: Dropout probability.
        out_dim: Output dimension (typically 1).
    """
    def __init__(
        self,
        num_feats=256,
        sigma=5.0,
        hidden_dim=256,
        num_blocks=8,
        act="silu",
        dropout=0.0,
        out_dim=1,
    ):
        super().__init__()
        self.ff = MultiScaleGaussianFourierFeatures(
            2,
            num_feats=num_feats,
            sigmas=(2.0, 6.0, sigma),
            seed=0,
        )

        self.in_proj = nn.Linear(2 * num_feats, hidden_dim)

        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, act=act, dropout=dropout)
            for _ in range(num_blocks)
        ])

        self.out_ln = nn.LayerNorm(hidden_dim)
        activation = nn.ReLU if act.lower() == "relu" else nn.SiLU
        self.out_act = activation()
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.ff(x)
        x = self.in_proj(x)
        x = self.blocks(x)
        x = self.out_act(self.out_ln(x))
        return self.out_proj(x)
    