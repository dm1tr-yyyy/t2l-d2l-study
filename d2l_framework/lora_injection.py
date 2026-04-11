"""
Динамическая инжекция LoRA в down_proj слои LLM.

Из reference lora_layer.py — через functools.partial:
  y = base_forward(x) + (x @ A.T) @ B.T * scaling

A и B сохраняют requires_grad через гиперсеть → backprop работает.
"""

from functools import partial
from operator import attrgetter

import torch
import torch.nn as nn

from .config import D2LConfig


def _lora_forward(
    x: torch.Tensor,
    A: torch.Tensor,        # [r, d_in]
    B: torch.Tensor,        # [r, d_out]
    scaling: float,
    original_forward,       # nn.Linear.forward
    self_module: nn.Module,  # the Linear module
    *args, **kwargs,
) -> torch.Tensor:
    """Patched forward: base + LoRA delta."""
    base_out = original_forward(x, *args, **kwargs)

    # LoRA: x @ A.T → [batch, seq, r] → @ B → [batch, seq, d_out]
    # A: [r, d_in], B: [r, d_out]
    delta = x.to(A.dtype) @ A.T  # [batch, seq, d_in] @ [d_in, r] → [batch, seq, r]
    delta = delta @ B             # [batch, seq, r] @ [r, d_out] → [batch, seq, d_out]
    delta = delta * scaling

    return base_out + delta.to(base_out.dtype)


def inject_lora(
    model: nn.Module,
    lora_dict: dict[str, torch.Tensor],
    config: D2LConfig,
) -> None:
    """
    Инжектит LoRA-веса в down_proj каждого слоя модели.

    Args:
        model: базовая LLM (e.g. Qwen3ForCausalLM)
        lora_dict: {"A": [batch, n_layers, r, d_in], "B": [batch, n_layers, r, d_out]}
        config: D2LConfig
    """
    A_all = lora_dict["A"][0]  # [n_layers, r, d_in] — drop batch dim
    B_all = lora_dict["B"][0]  # [n_layers, r, d_out]

    layers = model.model.layers  # Qwen: model.model.layers[i].mlp.down_proj

    for layer_idx in range(config.num_layers):
        module = layers[layer_idx].mlp.down_proj
        A = A_all[layer_idx]  # [r, d_in]
        B = B_all[layer_idx]  # [r, d_out]

        # Сохраняем оригинальный forward
        if not hasattr(module, "_original_forward"):
            module._original_forward = module.forward

        module.forward = partial(
            _lora_forward,
            A=A,
            B=B,
            scaling=config.lora_scaling,
            original_forward=module._original_forward,
            self_module=module,
        )
        module._lora_patched = True


def remove_lora(model: nn.Module, config: D2LConfig) -> None:
    """Убирает LoRA-патчи, восстанавливает оригинальные forward."""
    layers = model.model.layers

    for layer_idx in range(config.num_layers):
        module = layers[layer_idx].mlp.down_proj
        if hasattr(module, "_original_forward"):
            module.forward = module._original_forward
            del module._original_forward
            if hasattr(module, "_lora_patched"):
                del module._lora_patched


def is_lora_injected(model: nn.Module, config: D2LConfig) -> bool:
    """Проверяет есть ли активная LoRA-инжекция."""
    module = model.model.layers[0].mlp.down_proj
    return getattr(module, "_lora_patched", False)
