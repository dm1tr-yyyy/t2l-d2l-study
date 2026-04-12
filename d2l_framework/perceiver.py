"""
Perceiver Resampler для D2L.

Сжимает variable-length per-layer активации в фиксированный латентный вектор.
Архитектура из Idefics2PerceiverResampler, переписана с eager attention для MPS.

Поток:
  [batch, seq_len, hidden_size] →
  modality_projection (SwiGLU MLP) →
  [batch, seq_len, latent_size] →
  cross-attention blocks (Q из latents, KV из [context; latents]) →
  [batch, n_latent_queries, latent_size]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import D2LConfig


# ---------------------------------------------------------------------------
# SwiGLU MLP (как modality_projection в reference)
# ---------------------------------------------------------------------------

class SwiGLUMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.up_proj   = nn.Linear(input_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


# ---------------------------------------------------------------------------
# Cross-Attention: Q из latents, K/V из [context; latents]
# ---------------------------------------------------------------------------

class PerceiverAttention(nn.Module):
    def __init__(self, latent_size: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = latent_size // n_heads
        assert latent_size % n_heads == 0

        self.q_proj = nn.Linear(latent_size, latent_size, bias=False)
        self.k_proj = nn.Linear(latent_size, latent_size, bias=False)
        self.v_proj = nn.Linear(latent_size, latent_size, bias=False)
        self.o_proj = nn.Linear(latent_size, latent_size, bias=False)

    def forward(
        self,
        latents: torch.Tensor,  # [batch, n_queries, latent_size]
        context: torch.Tensor,  # [batch, seq_len, latent_size]
    ) -> torch.Tensor:
        bs = latents.shape[0]

        # Concat: KV from [context; latents]
        kv_input = torch.cat([context, latents], dim=1)

        q = self.q_proj(latents)
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)

        # [batch, seq, heads, head_dim] → [batch, heads, seq, head_dim]
        q = q.view(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # SDPA (MPS compatible)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.head_dim)

        return self.o_proj(attn_out)


# ---------------------------------------------------------------------------
# Perceiver Block: cross-attn + FFN + residual + RMSNorm
# ---------------------------------------------------------------------------

class PerceiverBlock(nn.Module):
    def __init__(self, latent_size: int, n_heads: int):
        super().__init__()
        self.attn_norm   = RMSNorm(latent_size)
        self.attn        = PerceiverAttention(latent_size, n_heads)
        self.ffn_norm    = RMSNorm(latent_size)
        self.ffn         = SwiGLUMLP(latent_size, latent_size * 4, latent_size)

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        # Cross-attention + residual
        latents = latents + self.attn(self.attn_norm(latents), context)
        # FFN + residual
        latents = latents + self.ffn(self.ffn_norm(latents))
        return latents


# ---------------------------------------------------------------------------
# Perceiver Resampler
# ---------------------------------------------------------------------------

class PerceiverResampler(nn.Module):
    """
    Input:  [batch, seq_len, hidden_size]  (one layer's activations)
    Output: [batch, n_latent_queries, latent_size]
    """

    def __init__(self, config: D2LConfig):
        super().__init__()
        self.config = config

        # Проекция из hidden_size модели в latent_size perceiver'а
        self.modality_proj = SwiGLUMLP(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size * 2,
            output_size=config.latent_size,
        )

        # Обучаемые латентные запросы
        self.latents = nn.Parameter(
            torch.randn(config.n_latent_queries, config.latent_size) * 0.02
        )

        # Cross-attention блоки
        self.blocks = nn.ModuleList([
            PerceiverBlock(config.latent_size, config.perceiver_heads)
            for _ in range(config.perceiver_blocks)
        ])

        self.final_norm = RMSNorm(config.latent_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch, seq_len, hidden_size]
        Returns:
            [batch, n_latent_queries, latent_size]
        """
        bs = features.shape[0]

        # Проецируем в latent space
        context = self.modality_proj(features)  # [batch, seq_len, latent_size]

        # Expand latents для batch
        latents = self.latents.unsqueeze(0).expand(bs, -1, -1)  # [batch, n_queries, latent_size]

        # Cross-attention blocks
        for block in self.blocks:
            latents = block(latents, context)

        return self.final_norm(latents)


if __name__ == "__main__":
    from .config import auto_config

    cfg = auto_config()
    perc = PerceiverResampler(cfg).to(cfg.device)

    n_params = sum(p.numel() for p in perc.parameters())
    print(f"Perceiver params: {n_params/1e6:.2f}M")

    x = torch.randn(1, 200, cfg.hidden_size, device=cfg.device)
    out = perc(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    # Expected: [1, 8, 512]
