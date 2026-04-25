"""
Обучающий цикл D2L гиперсети.

KL-дистилляция:
  Teacher: LLM + документ в промпте (frozen) → top-K logprobs
  Student: LLM + сгенерированный LoRA, без документа → logits
  Loss: sparse KL(teacher || student) + L1 reg

Запуск:
    uv run python -m d2l_framework.train
"""

import time
from pathlib import Path

from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import D2LConfig, auto_config
from .data import get_dataloader
from .doc_to_lora import DocToLoRA
from .lora_injection import inject_lora, remove_lora
from .losses import compute_teacher_topk, kl_distillation_loss, l1_regularization


def train(config: D2LConfig | None = None, resume_from: str | None = None, max_samples: int | None = None):
    if config is None:
        config = auto_config()

    device = torch.device(config.device)
    print(f"Device: {device}")
    print(f"Model:  {config.model_name}")

    # -------------------------------------------------------------------------
    # Модель и токенизатор
    # -------------------------------------------------------------------------
    print("\nЗагружаю базовую модель...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # bfloat16 на CUDA/A100, float32 на MPS/CPU (MPS не поддерживает bf16 везде)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    # -------------------------------------------------------------------------
    # Гиперсеть (переиспользует base_model как encoder)
    # -------------------------------------------------------------------------
    print("Строю гиперсеть...")
    d2l = DocToLoRA(config, base_model=base_model)
    d2l.to(device)
    # Perceiver + HyperLoRA → тот же dtype что и base_model (bf16 на CUDA)
    d2l.perceiver.to(dtype=dtype)
    d2l.hyperlora.to(dtype=dtype)

    if resume_from is not None:
        print(f"Загружаю чекпоинт: {resume_from}")
        d2l.load_checkpoint(resume_from)

    # Perceiver + HyperLoRA в train mode
    d2l.perceiver.train()
    d2l.hyperlora.train()

    trainable = d2l.num_trainable()
    print(f"Обучаемые параметры: {trainable/1e6:.1f}M")

    # -------------------------------------------------------------------------
    # Данные
    # -------------------------------------------------------------------------
    print("Загружаю SQuAD...")
    dataloader = get_dataloader(tokenizer, config, split="train", max_samples=max_samples)
    data_iter = iter(dataloader)

    # -------------------------------------------------------------------------
    # Оптимизатор
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(d2l.trainable_parameters(), lr=config.lr, weight_decay=0.01)

    warmup_steps = int(config.max_steps * config.warmup_ratio)

    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(config.max_steps - warmup_steps, 1)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    ckpt_dir = Path("d2l_checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    print(f"\nНачинаю обучение: {config.max_steps} шагов, grad_accum={config.grad_accum}")
    print(f"Effective batch = {config.batch_size * config.grad_accum}")
    print("=" * 60)

    optimizer.zero_grad()
    running_loss = 0.0
    log_interval = 50
    steps_per_epoch = len(dataloader)
    save_interval = steps_per_epoch // 2  # каждые полэпохи
    t0 = time.time()

    for step in range(1, config.max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.to(device) for k, v in batch.items()}

        # -------------------------------------------------------------------
        # 1. Teacher forward (с контекстом, no grad)
        # -------------------------------------------------------------------
        with torch.no_grad():
            teacher_out = base_model(
                input_ids=batch["teacher_input_ids"],
                attention_mask=batch["teacher_attention_mask"],
            )
            teacher_topk_lp, teacher_topk_idx = compute_teacher_topk(
                teacher_out.logits, batch["teacher_labels"], top_k=config.kl_top_k,
            )

        # -------------------------------------------------------------------
        # 2. Гиперсеть: документ → LoRA
        # -------------------------------------------------------------------
        lora_dict = d2l(batch["ctx_input_ids"], batch["ctx_attention_mask"])

        # -------------------------------------------------------------------
        # 3. Инжектим LoRA в base_model
        # -------------------------------------------------------------------
        inject_lora(base_model, lora_dict, config)

        # -------------------------------------------------------------------
        # 4. Student forward (без контекста, с LoRA)
        # -------------------------------------------------------------------
        student_out = base_model(
            input_ids=batch["student_input_ids"],
            attention_mask=batch["student_attention_mask"],
        )

        # -------------------------------------------------------------------
        # 5. Убираем LoRA
        # -------------------------------------------------------------------
        remove_lora(base_model, config)

        # -------------------------------------------------------------------
        # 6. Loss
        # -------------------------------------------------------------------
        kl_loss = kl_distillation_loss(
            student_out.logits, batch["student_labels"],
            teacher_topk_lp, teacher_topk_idx,
        )
        l1_loss = l1_regularization(lora_dict)
        loss = kl_loss + config.l1_reg * l1_loss

        # Grad accumulation
        loss = loss / config.grad_accum
        loss.backward()

        running_loss += loss.item()

        if step % config.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(d2l.trainable_parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # -------------------------------------------------------------------
        # Logging
        # -------------------------------------------------------------------
        if step % log_interval == 0:
            elapsed = time.time() - t0
            avg_loss = running_loss / log_interval * config.grad_accum
            lr_now = scheduler.get_last_lr()[0]

            elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))

            print(
                f"[{step:>6}/{config.max_steps}] "
                f"loss={avg_loss:.4f} "
                f"kl={kl_loss.item():.4f} "
                f"l1={l1_loss.item():.4f} "
                f"lr={lr_now:.2e} "
                f"({elapsed/step:.2f}s/step)"
                f"{datetime.now().strftime('%H:%M:%S')} - текущее время"
            )
            running_loss = 0.0

        # -------------------------------------------------------------------
        # Checkpointing
        # -------------------------------------------------------------------
        if step % save_interval == 0:
            path = ckpt_dir / f"step_{step}.pt"
            d2l.save_checkpoint(str(path))
            print(f"  → Saved {path}")

        # Очистка MPS кэша
        if step % 100 == 0 and device.type == "mps":
            torch.mps.empty_cache()

    # Final save
    final_path = ckpt_dir / "final.pt"
    d2l.save_checkpoint(str(final_path))
    print(f"\nОбучение завершено. Финальный чекпоинт: {final_path}")
    total_time = time.time() - t0
    print(f"Время: {total_time/60:.1f} мин ({total_time/config.max_steps:.2f}s/step)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train D2L hypernetwork")
    parser.add_argument("--model", default=None,
                        help="HuggingFace model name (default from D2LConfig)")
    parser.add_argument("--resume_from", default=None,
                        help="Path to checkpoint .pt to resume/fine-tune from")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max training steps")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=None,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit training samples (for quick tests)")
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("model", "resume_from", "max_samples") and v is not None}

    cfg = auto_config(args.model, **overrides) if args.model else auto_config(**overrides)
    train(cfg, resume_from=args.resume_from, max_samples=args.max_samples)
