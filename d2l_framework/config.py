"""
Конфигурация D2L гиперсети.
auto_config(model_name) извлекает размерности из HuggingFace config.
"""

from dataclasses import dataclass

import torch
from transformers import AutoConfig


@dataclass
class D2LConfig:
    # --- Базовая модель ---
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    num_layers: int = 0         # auto
    hidden_size: int = 0        # auto
    intermediate_size: int = 0  # auto
    target_module: str = "down_proj"

    # --- LoRA ---
    lora_r: int = 8
    lora_alpha: float = 0.0     # auto: r^(3/2) * 2

    # --- Perceiver (из статьи D2L, Listing 1) ---
    # Дефолты Idefics2PerceiverConfig: n_heads=16, head_dim=128, num_kv_heads=4
    # → GQA 16:4. q_proj 512→2048, k/v_proj 512→512, o_proj 2048→512.
    n_latent_queries: int = 8       # = lora_r
    perceiver_heads: int = 16
    perceiver_kv_heads: int = 4
    perceiver_head_dim: int = 128
    perceiver_blocks: int = 8       # статья: 8 cross-attention блоков
    latent_size: int = 512

    # --- HyperLoRA ---
    num_pre_head_layers: int = 1

    # --- Encoder ---
    max_chunk_len: int = 512    # для ctx токенизации (SQuAD: ~200 токенов)
    max_teacher_len: int = 1024 # для teacher prompt (ctx+Q+A)

    # --- Training ---
    lr: float = 3e-5
    # 2 эпохи на SQuAD: 87599 / batch_size=4 * 2 = 43800 шагов
    max_steps: int = 43800
    batch_size: int = 4
    grad_accum: int = 8         # effective batch = 32
    warmup_ratio: float = 0.05
    kl_top_k: int = 16
    l1_reg: float = 1e-4

    # --- Device ---
    device: str = ""  # auto

    @property
    def lora_scaling(self) -> float:
        return self.lora_alpha / self.lora_r

    @property
    def d_in(self) -> int:
        """Input dim для down_proj: intermediate_size."""
        return self.intermediate_size

    @property
    def d_out(self) -> int:
        """Output dim для down_proj: hidden_size."""
        return self.hidden_size


def auto_config(model_name: str = "Qwen/Qwen3-4B-Instruct-2507", **overrides) -> D2LConfig:
    """Создаёт конфиг, автоматически читая размерности из HF модели."""
    hf = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    cfg = D2LConfig(
        model_name=model_name,
        num_layers=hf.num_hidden_layers,
        hidden_size=hf.hidden_size,
        intermediate_size=hf.intermediate_size,
    )

    # lora_alpha = r^(3/2) * 2  (из reference repo, model_loading.py:178)
    cfg.lora_alpha = cfg.lora_r ** 1.5 * 2

    # Device
    if not cfg.device:
        if torch.cuda.is_available():
            cfg.device = "cuda"
        elif torch.backends.mps.is_available():
            cfg.device = "mps"
        else:
            cfg.device = "cpu"

    # Overrides
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    return cfg


if __name__ == "__main__":
    cfg = auto_config()
    print(f"Model:        {cfg.model_name}")
    print(f"Layers:       {cfg.num_layers}")
    print(f"Hidden:       {cfg.hidden_size}")
    print(f"Intermediate: {cfg.intermediate_size}")
    print(f"LoRA r={cfg.lora_r}, alpha={cfg.lora_alpha:.2f}, scaling={cfg.lora_scaling:.4f}")
    print(f"d_in={cfg.d_in}, d_out={cfg.d_out}, d_lora={cfg.d_in + cfg.d_out}")
    print(f"Perceiver:    {cfg.n_latent_queries} queries, {cfg.perceiver_blocks} blocks, "
          f"{cfg.perceiver_heads}q/{cfg.perceiver_kv_heads}kv heads × "
          f"{cfg.perceiver_head_dim} head_dim, latent={cfg.latent_size}")
    print(f"q_proj:       {cfg.latent_size} → {cfg.perceiver_heads * cfg.perceiver_head_dim}")
    print(f"k/v_proj:     {cfg.latent_size} → {cfg.perceiver_kv_heads * cfg.perceiver_head_dim}")
    print(f"Device:       {cfg.device}")
    print(f"Batch:        {cfg.batch_size} × grad_accum {cfg.grad_accum} = {cfg.batch_size * cfg.grad_accum} effective")
