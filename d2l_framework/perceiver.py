"""
Perceiver Resampler для D2L.

Архитектура из статьи D2L (Listing 1) — Idefics2PerceiverResampler:
  - modality_projection: SwiGLU MLP (hidden_size → latent_size)
  - 8 cross-attention блоков с 5 RMSNorm на блок
  - MQA: 4 query heads (512-dim each) → q_proj 512→2048,
         1 kv head (512-dim) → k/v_proj 512→512
  - o_proj: 2048→512

Поток:
  [batch, seq_len, hidden_size] →
  modality_projection →
  [batch, seq_len, latent_size] →
  8 × cross-attention block (Q из latents, KV из [context; latents]) →
  [batch, n_latent_queries, latent_size]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import D2LConfig


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class SwiGLUMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.up_proj   = nn.Linear(input_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


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
# Cross-Attention (MQA): 4 query heads × 512-dim, 1 kv head × 512-dim
# Соответствует Listing 1: q_proj 512→2048, k/v 512→512, o_proj 2048→512
# ---------------------------------------------------------------------------

class PerceiverAttention(nn.Module):
    def __init__(self, latent_size: int, n_heads: int):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = latent_size   # каждая query-голова имеет full latent_size dims

        # MQA: n_heads query heads, 1 kv head
        self.q_proj = nn.Linear(latent_size, latent_size * n_heads, bias=False)
        self.k_proj = nn.Linear(latent_size, latent_size, bias=False)
        self.v_proj = nn.Linear(latent_size, latent_size, bias=False)
        self.o_proj = nn.Linear(latent_size * n_heads, latent_size, bias=False)

    def forward(
        self,
        latents: torch.Tensor,  # [batch, n_queries, latent_size]  — уже нормализованы
        context: torch.Tensor,  # [batch, seq_len, latent_size]    — уже нормализованы
    ) -> torch.Tensor:
        bs, n_q, _ = latents.shape

        # KV: объединяем context и latents (как в reference Idefics2)
        kv = torch.cat([context, latents], dim=1)  # [b, seq+n_q, latent]
        n_kv = kv.shape[1]

        q = self.q_proj(latents)  # [b, n_q, latent * n_heads]
        k = self.k_proj(kv)       # [b, n_kv, latent]
        v = self.v_proj(kv)       # [b, n_kv, latent]

        # Reshape для multi-head attention
        q = q.view(bs, n_q, self.n_heads, self.head_dim).transpose(1, 2)
        # [b, n_heads, n_q, head_dim]

        # KV: 1 head, expand до n_heads для SDPA
        k = k.unsqueeze(1).expand(bs, self.n_heads, n_kv, self.head_dim)
        v = v.unsqueeze(1).expand(bs, self.n_heads, n_kv, self.head_dim)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        # [b, n_heads, n_q, head_dim]

        attn_out = attn_out.transpose(1, 2).contiguous().view(bs, n_q, self.n_heads * self.head_dim)
        return self.o_proj(attn_out)


# ---------------------------------------------------------------------------
# Perceiver Block (Idefics2PerceiverLayer из Listing 1)
# 5 RMSNorm: input_latents, input_context, post_attn, pre_ff, post_ff
# ---------------------------------------------------------------------------

class PerceiverBlock(nn.Module):
    def __init__(self, latent_size: int, n_heads: int):
        super().__init__()
        self.input_latents_norm  = RMSNorm(latent_size)
        self.input_context_norm  = RMSNorm(latent_size)
        self.attn                = PerceiverAttention(latent_size, n_heads)
        self.post_attn_norm      = RMSNorm(latent_size)
        self.pre_ff_norm         = RMSNorm(latent_size)
        self.post_ff_norm        = RMSNorm(latent_size)
        self.ffn                 = SwiGLUMLP(latent_size, latent_size * 4, latent_size)

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        # Cross-attention: pre-norm latents и context, post-norm output + residual
        residual = latents
        attn_out = self.attn(
            self.input_latents_norm(latents),
            self.input_context_norm(context),
        )
        latents = residual + self.post_attn_norm(attn_out)

        # FFN: pre-norm input, post-norm output + residual
        residual = latents
        ffn_out  = self.ffn(self.pre_ff_norm(latents))
        latents  = residual + self.post_ff_norm(ffn_out)

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

        # Проекция из hidden_size модели в latent_size perceiver'а (SwiGLU MLP)
        self.modality_proj = SwiGLUMLP(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size * 4,
            output_size=config.latent_size,
        )

        # Обучаемые латентные запросы
        self.latents = nn.Parameter(
            torch.randn(config.n_latent_queries, config.latent_size) * 0.02
        )

        # 8 cross-attention блоков (статья: 8 blocks without self-attention)
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

        context = self.modality_proj(features)  # [batch, seq_len, latent_size]
        latents = self.latents.unsqueeze(0).expand(bs, -1, -1)

        for block in self.blocks:
            latents = block(latents, context)

        return self.final_norm(latents)


if __name__ == "__main__":
    from .config import auto_config

    cfg = auto_config()
    perc = PerceiverResampler(cfg).to(cfg.device)

    n_params = sum(p.numel() for p in perc.parameters())
    print(f"Perceiver params: {n_params/1e6:.2f}M")
    print(f"q_proj: {cfg.latent_size} → {cfg.latent_size * cfg.perceiver_heads}  "
          f"(n_heads={cfg.perceiver_heads}, head_dim={cfg.latent_size})")

    x = torch.randn(1, 100, cfg.hidden_size, device=cfg.device)
    out = perc(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")  # Expected: [1, 8, 512]
