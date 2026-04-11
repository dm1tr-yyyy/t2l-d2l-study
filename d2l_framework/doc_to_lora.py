"""
DocToLoRA: сборка гиперсети.

encoder (frozen LLM) → per-layer perceiver → HyperLoRA heads → LoRA weights.

Документ → LoRA state_dict, готовый к инжекции в ту же LLM.
"""

import torch
import torch.nn as nn

from .config import D2LConfig
from .context_encoder import ContextEncoder
from .perceiver import PerceiverResampler
from .hyperlora import HyperLoRA


class DocToLoRA(nn.Module):
    """
    forward(input_ids, attention_mask) → {"A": [1, n_layers, r, d_in], "B": [1, n_layers, r, d_out]}

    Только perceiver + hyperlora обучаются. Encoder (LLM) заморожен.
    """

    def __init__(self, config: D2LConfig, base_model: nn.Module | None = None):
        super().__init__()
        self.config = config

        # Encoder: переиспользуем base_model если дан (экономия ~2.4 GB)
        self.encoder = ContextEncoder(config, base_model=base_model)

        # Shared perceiver: один для всех слоёв (layer-to-layer mode)
        self.perceiver = PerceiverResampler(config)

        # HyperLoRA heads
        self.hyperlora = HyperLoRA(config)

    @torch.no_grad()
    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encoder (no grad): [batch, seq] → [batch, num_layers, seq, hidden]."""
        return self.encoder(input_ids, attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq_len] — токены документа
            attention_mask: [batch, seq_len]
        Returns:
            {"A": [batch, n_layers, r, d_in], "B": [batch, n_layers, r, d_out]}
        """
        # 1. Per-layer activations (no grad)
        activations = self._encode(input_ids, attention_mask)
        # activations: [batch, num_layers, seq_len, hidden_size]

        bs, n_layers = activations.shape[:2]

        # 2. Per-layer perceiver: прогоняем каждый слой через shared perceiver
        layer_latents = []
        for layer_idx in range(n_layers):
            layer_act = activations[:, layer_idx]  # [batch, seq_len, hidden_size]
            latent = self.perceiver(layer_act)      # [batch, n_queries, latent_size]
            layer_latents.append(latent)

        # Stack: [batch, n_layers, n_queries, latent_size]
        # n_queries == lora_r (8 запросов → 8 рангов)
        stacked = torch.stack(layer_latents, dim=1)

        # 3. HyperLoRA: latents → A, B weights
        lora_dict = self.hyperlora(stacked)

        return lora_dict

    def trainable_parameters(self):
        """Параметры для оптимизатора (всё кроме encoder)."""
        for p in self.perceiver.parameters():
            yield p
        for p in self.hyperlora.parameters():
            yield p

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def save_checkpoint(self, path: str):
        """Сохраняет только обучаемые веса."""
        state = {
            "perceiver": self.perceiver.state_dict(),
            "hyperlora": self.hyperlora.state_dict(),
            "config": self.config.__dict__,
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location="cpu")
        self.perceiver.load_state_dict(state["perceiver"])
        self.hyperlora.load_state_dict(state["hyperlora"])


if __name__ == "__main__":
    from transformers import AutoTokenizer

    from .config import auto_config

    cfg = auto_config()
    print(f"Building DocToLoRA for {cfg.model_name}...")

    d2l = DocToLoRA(cfg)
    d2l.to(cfg.device)

    print(f"Total params:     {sum(p.numel() for p in d2l.parameters())/1e6:.1f}M")
    print(f"Trainable params: {d2l.num_trainable()/1e6:.1f}M")

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    doc = "The Aurora research station was built in 2019 on the coast of Greenland."
    inputs = tok(doc, return_tensors="pt", truncation=True, max_length=cfg.max_chunk_len)
    inputs = {k: v.to(cfg.device) for k, v in inputs.items()}

    lora = d2l(inputs["input_ids"], inputs["attention_mask"])
    print(f"\nDocument: '{doc}'")
    print(f"A: {lora['A'].shape}")
    print(f"B: {lora['B'].shape}")
