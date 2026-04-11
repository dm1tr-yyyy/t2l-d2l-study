"""
Контекстный энкодер: замороженная LLM, собирающая per-layer активации.

Из reference D2L — PerLayerActivations:
  - Убираем lm_head (нужны только hidden states)
  - output_hidden_states=True → stack по слоям
  - Chunking для длинных документов
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .config import D2LConfig


class ContextEncoder(nn.Module):
    """Замороженная LLM, возвращающая per-layer hidden states."""

    def __init__(self, config: D2LConfig, base_model: nn.Module | None = None):
        super().__init__()
        self.config = config

        if base_model is not None:
            # Переиспользуем модель (экономия памяти)
            self.model = base_model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float32,
                attn_implementation="eager",
            )

        # Замораживаем все параметры
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        self.max_chunk_len = config.max_chunk_len

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
        Returns:
            activations: [batch, num_layers, seq_len, hidden_size]
        """
        seq_len = input_ids.shape[1]

        if seq_len <= self.max_chunk_len:
            return self._forward_chunk(input_ids, attention_mask)

        # Chunking для длинных документов
        chunks = []
        for start in range(0, seq_len, self.max_chunk_len):
            end = min(start + self.max_chunk_len, seq_len)
            chunk_ids = input_ids[:, start:end]
            chunk_mask = attention_mask[:, start:end] if attention_mask is not None else None
            chunks.append(self._forward_chunk(chunk_ids, chunk_mask))

        # Конкатенируем по seq_len
        return torch.cat(chunks, dim=2)

    def _forward_chunk(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # hidden_states: tuple of (num_layers+1) tensors [batch, seq, hidden]
        # [0] — embedding output, [1..N] — layer outputs
        # Берём все слои кроме embedding (как в reference)
        layer_outputs = outputs.hidden_states[1:]  # num_layers tensors
        return torch.stack(layer_outputs, dim=1)  # [batch, num_layers, seq, hidden]


if __name__ == "__main__":
    from .config import auto_config

    cfg = auto_config()
    print(f"Loading {cfg.model_name}...")
    enc = ContextEncoder(cfg)
    enc.to(cfg.device)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    inputs = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt").to(cfg.device)

    out = enc(inputs["input_ids"], inputs["attention_mask"])
    print(f"Input:  {inputs['input_ids'].shape}")
    print(f"Output: {out.shape}")
    # Expected: [1, 28, seq_len, 1024]
