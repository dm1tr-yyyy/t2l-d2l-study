"""
HyperLoRA: генерация LoRA-весов (A, B) из латентного представления.

Из reference D2L (hypernet.py + lora_merger.combine_lora):
  - ResMLPBlock для предобработки
  - L2-нормализация
  - Per-layer linear head (заменяем einops.EinMix на nn.Parameter + einsum)
  - Learnable bias_A, bias_B, scaler_A, scaler_B
  - Bias стекается как ДОПОЛНИТЕЛЬНЫЕ r rank-строк к A/B → итоговый rank=2r
    (delta = x A_genᵀ B_gen + x bias_Aᵀ bias_B — два независимых LoRA параллельно)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import D2LConfig


# ---------------------------------------------------------------------------
# ResMLPBlock (pre-head processing)
# ---------------------------------------------------------------------------

class ResMLPBlock(nn.Module):
    def __init__(self, size: int, hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(size),
            nn.Linear(size, hidden),
            nn.SiLU(),
            nn.Linear(hidden, size),
            nn.LayerNorm(size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


# ---------------------------------------------------------------------------
# HyperLoRA
# ---------------------------------------------------------------------------

class HyperLoRA(nn.Module):
    """
    Input:  [batch, n_layers, lora_r, latent_size]  (per-layer perceiver output)
    Output: dict {"A": [batch, n_layers, r, d_in], "B": [batch, n_layers, r, d_out]}
    """

    def __init__(self, config: D2LConfig):
        super().__init__()
        self.config = config
        n_layers    = config.num_layers
        r           = config.lora_r
        d_latent    = config.latent_size
        d_in        = config.d_in   # intermediate_size (3072)
        d_out       = config.d_out  # hidden_size (1024)
        d_lora      = d_in + d_out  # head output: split into A and B

        # Pre-head ResMLPBlocks
        self.pre_head = nn.Sequential(*[
            ResMLPBlock(d_latent, d_latent * 4)
            for _ in range(config.num_pre_head_layers)
        ])

        # Per-layer linear head: [n_layers, latent_size, d_lora]
        # Заменяет einops EinMix — отдельные веса для каждого слоя.
        # Init из reference (hypernet.py:580-588, _bias_hyper_init):
        #   std = 0.5 / sqrt(d_latent + d_lora * r)
        # Деление на d_lora*r компенсирует большой выходной d_lora и масштаб через r.
        head_std = 0.5 / math.sqrt(d_latent + d_lora * r)
        self.head_weight = nn.Parameter(
            torch.randn(n_layers, d_latent, d_lora) * head_std
        )

        # Learnable biases для A и B (из reference, строки 276-293)
        # A: Gaussian init, B: zero init
        self.bias_A = nn.Parameter(
            torch.normal(0, 0.2 / (d_in * r) ** 0.5, (n_layers, r, d_in))
        )
        self.bias_B = nn.Parameter(
            torch.zeros(n_layers, r, d_out)
        )

        # Learnable scalers (из reference, строки 295-306)
        # scaler_A=1: A сразу несёт документо-специфичную информацию
        # scaler_B=0: LoRA=0 на старте (delta = x@A.T@B, B=0 → delta=0)
        self.scaler_A = nn.Parameter(torch.ones(1, n_layers, r, 1))
        self.scaler_B = nn.Parameter(torch.zeros(1, n_layers, r, 1))

    def forward(
        self, latents: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            latents: [batch, n_layers, lora_r, latent_size]
        Returns:
            {"A": [batch, n_layers, 2r, d_in], "B": [batch, n_layers, 2r, d_out]}
            Первые r строк — generated (× scaler), вторые r — bias (doc-agnostic).
        """
        d_in  = self.config.d_in
        d_out = self.config.d_out

        # Pre-head processing
        latents = self.pre_head(latents)  # [batch, n_layers, r, latent_size]

        # L2 normalize (из reference, строки 422-423)
        latents = F.normalize(latents, dim=-1)

        # Per-layer linear head: einsum("b l r d, l d o -> b l r o")
        flat = torch.einsum("blrd,ldo->blro", latents, self.head_weight)
        # flat: [batch, n_layers, r, d_in + d_out]

        # Split into A and B
        A = flat[..., :d_in]   # [batch, n_layers, r, d_in]
        B = flat[..., d_in:]   # [batch, n_layers, r, d_out]

        # Apply scalers (без сложения с bias — bias стекается ниже как rank-строки)
        A = A * self.scaler_A
        B = B * self.scaler_B

        # Стек bias-а как доп. r rank-строк (reference combine_lora):
        # эффект: delta = x A_genᵀ B_gen + x bias_Aᵀ bias_B (две независимые LoRA)
        bs = A.shape[0]
        bias_A = self.bias_A.unsqueeze(0).expand(bs, -1, -1, -1)  # [batch, L, r, d_in]
        bias_B = self.bias_B.unsqueeze(0).expand(bs, -1, -1, -1)  # [batch, L, r, d_out]

        A = torch.cat([A, bias_A], dim=2)  # [batch, L, 2r, d_in]
        B = torch.cat([B, bias_B], dim=2)  # [batch, L, 2r, d_out]

        return {"A": A, "B": B}


if __name__ == "__main__":
    from .config import auto_config

    cfg = auto_config()
    hyper = HyperLoRA(cfg).to(cfg.device)

    n = sum(p.numel() for p in hyper.parameters())
    print(f"HyperLoRA params: {n/1e6:.2f}M")

    # Simulate perceiver output for all layers
    x = torch.randn(1, cfg.num_layers, cfg.lora_r, cfg.latent_size, device=cfg.device)
    out = hyper(x)
    print(f"Input:  {x.shape}")
    print(f"A:      {out['A'].shape}  (expected [1, {cfg.num_layers}, {cfg.lora_r}, {cfg.d_in}])")
    print(f"B:      {out['B'].shape}  (expected [1, {cfg.num_layers}, {cfg.lora_r}, {cfg.d_out}])")
