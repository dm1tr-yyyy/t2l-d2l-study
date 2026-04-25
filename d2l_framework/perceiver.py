"""
Perceiver Resampler для D2L.

Архитектура из статьи D2L (Listing 1) — Idefics2Perceiver:
  - modality_projection: SwiGLU MLP (hidden_size → latent_size)
  - encoder: N × Idefics2PerceiverLayer (cross-attn, 5 RMSNorm на блок)
  - decoder: 1 × Idefics2PerceiverLayer (cross-attn над выходом encoder'а
             со своим latents_q; даёт финальные r латентов)
  - MQA: 4 query heads (512-dim each) → q_proj 512→2048,
         1 kv head (512-dim) → k/v_proj 512→512
  - o_proj: 2048→512

Поток:
  [batch, seq_len, hidden_size]
    → modality_projection
    → [batch, seq_len, latent_size]
    → encoder (N cross-attn) → [batch, n_latents, latent_size]
    → decoder (1 cross-attn, свои latents_q) → [batch, n_latents, latent_size]
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
# Cross-Attention (GQA): n_heads query heads, n_kv_heads kv heads, head_dim each.
# Дефолты reference Idefics2PerceiverConfig: 16q / 4kv × 128 head_dim.
# При latent=512: q_proj 512→16·128=2048, k/v_proj 512→4·128=512, o_proj 2048→512.
#
# KV вычисляется ТОЛЬКО из context (FlashAttention2 ветка reference,
# idefics2.py:357-447 при is_cross_attn=True). Latents в KV не подмешиваются.
# ---------------------------------------------------------------------------

class PerceiverAttention(nn.Module):
    def __init__(
        self,
        latent_size: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim   = head_dim
        self.kv_groups  = n_heads // n_kv_heads

        self.q_proj = nn.Linear(latent_size, n_heads    * head_dim, bias=False)
        self.k_proj = nn.Linear(latent_size, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(latent_size, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, latent_size,    bias=False)

    def forward(
        self,
        latents: torch.Tensor,  # [batch, n_queries, latent_size]
        context: torch.Tensor,  # [batch, ctx_len,   latent_size]
    ) -> torch.Tensor:
        bs, n_q, _ = latents.shape
        n_kv = context.shape[1]

        q = self.q_proj(latents)  # [b, n_q,  n_heads    * head_dim]
        k = self.k_proj(context)  # [b, n_kv, n_kv_heads * head_dim]
        v = self.v_proj(context)  # [b, n_kv, n_kv_heads * head_dim]

        # [b, n_heads, n_q, head_dim]
        q = q.view(bs, n_q, self.n_heads, self.head_dim).transpose(1, 2)
        # [b, n_kv_heads, n_kv, head_dim]
        k = k.view(bs, n_kv, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, n_kv, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # GQA: повторяем каждую kv-голову kv_groups раз вдоль head-оси (matches repeat_kv в reference)
        k = k.repeat_interleave(self.kv_groups, dim=1)  # [b, n_heads, n_kv, head_dim]
        v = v.repeat_interleave(self.kv_groups, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        # [b, n_heads, n_q, head_dim]

        attn_out = attn_out.transpose(1, 2).contiguous().view(bs, n_q, self.n_heads * self.head_dim)
        return self.o_proj(attn_out)


# ---------------------------------------------------------------------------
# Perceiver Block (Idefics2PerceiverLayer из Listing 1)
# 5 RMSNorm: input_latents, input_context, post_attn, pre_ff, post_ff
# ---------------------------------------------------------------------------

class PerceiverBlock(nn.Module):
    def __init__(
        self,
        latent_size: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.input_latents_norm  = RMSNorm(latent_size)
        self.input_context_norm  = RMSNorm(latent_size)
        self.attn                = PerceiverAttention(latent_size, n_heads, n_kv_heads, head_dim)
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
# PerceiverStack: один стек latents_q + N × cross-attn block + final RMSNorm
# Соответствует Idefics2PerceiverResampler в reference (без modality_projection)
# ---------------------------------------------------------------------------

class PerceiverStack(nn.Module):
    """
    Input:  [batch, ctx_len, latent_size]  — context (уже в latent space)
    Output: [batch, n_latents, latent_size]
    """

    def __init__(
        self,
        latent_size: int,
        n_latents: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        num_blocks: int,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, latent_size) * 0.02)
        self.blocks = nn.ModuleList([
            PerceiverBlock(latent_size, n_heads, n_kv_heads, head_dim)
            for _ in range(num_blocks)
        ])
        self.final_norm = RMSNorm(latent_size)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        bs = context.shape[0]
        latents = self.latents.unsqueeze(0).expand(bs, -1, -1)
        for block in self.blocks:
            latents = block(latents, context)
        return self.final_norm(latents)


# ---------------------------------------------------------------------------
# Perceiver Resampler (Idefics2Perceiver из Listing 1):
#   modality_projection → encoder (N cross-attn) → decoder (1 cross-attn)
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

        # Encoder: N cross-attention блоков
        self.encoder = PerceiverStack(
            latent_size=config.latent_size,
            n_latents=config.n_latent_queries,
            n_heads=config.perceiver_heads,
            n_kv_heads=config.perceiver_kv_heads,
            head_dim=config.perceiver_head_dim,
            num_blocks=config.perceiver_blocks,
        )

        # Decoder: 1 cross-attention блок над выходом encoder'а со своим latents_q
        self.decoder = PerceiverStack(
            latent_size=config.latent_size,
            n_latents=config.n_latent_queries,
            n_heads=config.perceiver_heads,
            n_kv_heads=config.perceiver_kv_heads,
            head_dim=config.perceiver_head_dim,
            num_blocks=1,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch, seq_len, hidden_size]
        Returns:
            [batch, n_latent_queries, latent_size]
        """
        projected = self.modality_proj(features)   # [batch, seq, latent_size]
        encoder_out = self.encoder(projected)      # [batch, n_latents, latent_size]
        decoder_out = self.decoder(encoder_out)    # [batch, n_latents, latent_size]
        return decoder_out


if __name__ == "__main__":
    from .config import auto_config

    cfg = auto_config()
    perc = PerceiverResampler(cfg).to(cfg.device)

    n_params = sum(p.numel() for p in perc.parameters())
    print(f"Perceiver params: {n_params/1e6:.2f}M")
    q_dim = cfg.perceiver_heads * cfg.perceiver_head_dim
    kv_dim = cfg.perceiver_kv_heads * cfg.perceiver_head_dim
    print(f"q_proj:   {cfg.latent_size} → {q_dim}  "
          f"(n_heads={cfg.perceiver_heads}, head_dim={cfg.perceiver_head_dim})")
    print(f"k/v_proj: {cfg.latent_size} → {kv_dim}  (n_kv_heads={cfg.perceiver_kv_heads})")

    x = torch.randn(1, 100, cfg.hidden_size, device=cfg.device)
    out = perc(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")  # Expected: [1, 8, 512]
