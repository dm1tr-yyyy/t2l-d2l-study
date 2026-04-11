"""
HyperLoRA: генерация LoRA-весов (A, B) из латентного представления.

Из reference D2L (hypernet.py):
  - ResMLPBlock для предобработки
  - L2-нормализация
  - Per-layer linear head (заменяем einops.EinMix на nn.Parameter + einsum)
  - Learnable bias_A, bias_B, scaler_A, scaler_B
"""

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
        self.norm = nn.LayerNorm(size)
        self.fc1 = nn.Linear(size, hidden)
        self.fc2 = nn.Linear(hidden, size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.fc2(h)
        return x + h


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
        # Заменяет einops EinMix — отдельные веса для каждого слоя
        self.head_weight = nn.Parameter(
            torch.randn(n_layers, d_latent, d_lora) * (1.0 / d_latent ** 0.5)
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
        self.scaler_A = nn.Parameter(torch.ones(1, n_layers, r, 1))
        self.scaler_B = nn.Parameter(torch.zeros(1, n_layers, r, 1))

    def forward(
        self, latents: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            latents: [batch, n_layers, lora_r, latent_size]
        Returns:
            {"A": [batch, n_layers, r, d_in], "B": [batch, n_layers, r, d_out]}
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

        # Apply scalers and add biases
        A = A * self.scaler_A + self.bias_A  # broadcast over batch
        B = B * self.scaler_B + self.bias_B

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
