"""
Подбор batch_size для D2L на текущем GPU.

Запуск:
    uv run python benchmark_batch.py

Делает пробные прогоны training step с разными batch_size,
меряет пиковое потребление VRAM и рекомендует конфигурацию.
"""

import gc
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from d2l_framework.config import auto_config
from d2l_framework.doc_to_lora import DocToLoRA
from d2l_framework.lora_injection import inject_lora, remove_lora
from d2l_framework.losses import compute_teacher_topk, kl_distillation_loss, l1_regularization


def gpu_info():
    if not torch.cuda.is_available():
        print("CUDA недоступна")
        return
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / 1024**3
        print(f"GPU {i}: {props.name}  {total:.1f} GB VRAM")


def reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def mem_used_gb() -> float:
    return torch.cuda.memory_reserved() / 1024**3


def peak_mem_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**3


def make_fake_batch(batch_size: int, config, device, tokenizer):
    """Синтетический батч с реалистичными длинами."""
    pad = tokenizer.pad_token_id

    def rand_ids(bs, length):
        ids = torch.randint(100, tokenizer.vocab_size - 100, (bs, length), device=device)
        return ids

    def rand_labels(bs, length):
        # ~30% токенов размечены как ответ, остальные -100
        labels = torch.full((bs, length), -100, device=device)
        start = length * 7 // 10
        labels[:, start:] = torch.randint(100, tokenizer.vocab_size - 100, (bs, length - start), device=device)
        return labels

    ctx_len     = 256   # типичный SQuAD context
    teacher_len = 512   # context + question + template overhead
    student_len = 64    # только вопрос

    return {
        "ctx_input_ids":          rand_ids(batch_size, ctx_len),
        "ctx_attention_mask":     torch.ones(batch_size, ctx_len, dtype=torch.long, device=device),
        "teacher_input_ids":      rand_ids(batch_size, teacher_len),
        "teacher_attention_mask": torch.ones(batch_size, teacher_len, dtype=torch.long, device=device),
        "teacher_labels":         rand_labels(batch_size, teacher_len),
        "student_input_ids":      rand_ids(batch_size, student_len),
        "student_attention_mask": torch.ones(batch_size, student_len, dtype=torch.long, device=device),
        "student_labels":         rand_labels(batch_size, student_len),
    }


def trial_step(batch_size: int, base_model, d2l, config, device, tokenizer) -> tuple[bool, float, float]:
    """
    Один пробный шаг. Возвращает (ok, peak_gb, sec_per_step).
    """
    reset_memory()
    try:
        batch = make_fake_batch(batch_size, config, device, tokenizer)

        t0 = time.perf_counter()

        # Teacher
        with torch.no_grad():
            teacher_out = base_model(
                input_ids=batch["teacher_input_ids"],
                attention_mask=batch["teacher_attention_mask"],
            )
            teacher_topk_lp, teacher_topk_idx = compute_teacher_topk(
                teacher_out.logits, batch["teacher_labels"], top_k=config.kl_top_k,
            )

        # Hypernetwork → LoRA
        lora_dict = d2l(batch["ctx_input_ids"], batch["ctx_attention_mask"])
        inject_lora(base_model, lora_dict, config)

        # Student forward + backward
        student_out = base_model(
            input_ids=batch["student_input_ids"],
            attention_mask=batch["student_attention_mask"],
        )
        remove_lora(base_model, config)

        kl = kl_distillation_loss(
            student_out.logits, batch["student_labels"],
            teacher_topk_lp, teacher_topk_idx,
        )
        l1 = l1_regularization(lora_dict)
        loss = kl + config.l1_reg * l1
        loss.backward()

        dt = time.perf_counter() - t0
        peak = peak_mem_gb()

        # Очистка
        for p in d2l.trainable_parameters():
            if p.grad is not None:
                p.grad = None

        return True, peak, dt

    except torch.cuda.OutOfMemoryError:
        remove_lora(base_model, config)
        for p in d2l.trainable_parameters():
            if p.grad is not None:
                p.grad = None
        reset_memory()
        return False, 0.0, 0.0


def main():
    print("=" * 60)
    print("D2L Batch Size Benchmark")
    print("=" * 60)
    gpu_info()

    if not torch.cuda.is_available():
        print("Нет CUDA — бенчмарк только для GPU.")
        return

    device = torch.device("cuda")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\nTotal VRAM: {total_vram:.1f} GB")

    config = auto_config()
    dtype  = torch.bfloat16

    print(f"\nЗагружаю {config.model_name} (bf16)...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    model_mem = torch.cuda.memory_reserved() / 1024**3
    print(f"Модель загружена: {model_mem:.1f} GB")

    print("Строю гиперсеть...")
    d2l = DocToLoRA(config, base_model=base_model)
    d2l.perceiver.to(device=device, dtype=dtype)
    d2l.hyperlora.to(device=device, dtype=dtype)
    d2l.perceiver.train()
    d2l.hyperlora.train()

    base_mem = torch.cuda.memory_reserved() / 1024**3
    free_mem = total_vram - base_mem
    print(f"После гиперсети: {base_mem:.1f} GB  (свободно ~{free_mem:.1f} GB)")
    print(f"Параметры: {d2l.num_trainable()/1e6:.1f}M обучаемых")

    # Тестируем batch sizes
    candidates = [1, 2, 4, 6, 8, 12, 16]
    print(f"\n{'batch':>6}  {'peak GB':>8}  {'s/step':>8}  {'статус'}")
    print("-" * 40)

    results = []
    for bs in candidates:
        ok, peak, dt = trial_step(bs, base_model, d2l, config, device, tokenizer)
        status = "OK" if ok else "OOM"
        if ok:
            print(f"{bs:>6}  {peak:>8.1f}  {dt:>8.2f}  {status}")
            results.append((bs, peak, dt))
        else:
            print(f"{bs:>6}  {'—':>8}  {'—':>8}  {status}")
            break  # дальше нет смысла

    if not results:
        print("\nДаже batch_size=1 не влезает!")
        return

    # Рекомендация: берём максимальный batch что влез, с запасом ~10%
    max_bs, max_peak, max_dt = results[-1]
    # Если последний близко к лимиту — берём предпоследний
    safe_bs = max_bs if max_peak < total_vram * 0.90 else (results[-2][0] if len(results) > 1 else max_bs)

    # Effective batch = 32 (как в статье)
    target_effective = 32
    grad_accum = max(1, target_effective // safe_bs)
    effective   = safe_bs * grad_accum

    steps_per_epoch = 87599 // safe_bs
    max_steps_2ep   = steps_per_epoch * 2

    print(f"\n{'='*60}")
    print(f"РЕКОМЕНДАЦИЯ")
    print(f"{'='*60}")
    print(f"  batch_size   = {safe_bs}")
    print(f"  grad_accum   = {grad_accum}  (effective batch = {effective})")
    print(f"  max_steps    = {max_steps_2ep}  (2 эпохи SQuAD)")
    print(f"  peak VRAM    ≈ {results[-1][1]:.1f} GB / {total_vram:.0f} GB")
    print(f"  ~s/step      ≈ {results[-1][2]:.2f}s")
    est_hours = max_steps_2ep * results[-1][2] / 3600
    print(f"  ~время 2 эп  ≈ {est_hours:.1f}ч")
    print(f"\n  Запуск:")
    print(f"  uv run python -m d2l_framework.train \\")
    print(f"      --batch_size {safe_bs} --grad_accum {grad_accum} --max_steps {max_steps_2ep}")


if __name__ == "__main__":
    main()
