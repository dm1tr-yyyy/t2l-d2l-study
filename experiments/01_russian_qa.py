"""
Эксперимент 1: Русский язык — QA (SberSQuAD)
=============================================
Сравниваем три варианта Qwen3-0.6B:
  (1) Без контекста  — только вопрос, без параграфа
  (2) С контекстом   — вопрос + параграф в промпте
  (3) LoRA           — дообученная модель (с параграфом)

Это показывает, что умеет делать LoRA-адаптер, который гиперсеть
(Text-to-LoRA) генерирует автоматически по описанию задачи.

Запуск:
    uv run experiments/01_russian_qa.py

Результаты сохраняются в experiments/results/russian_qa.json
"""

import json
import re
import string
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

MODEL_NAME   = "Qwen/Qwen3-0.6B"
DATASET_NAME = "ERmak1581/QA_sberquad"   # колонки: Q, C, A; только train-сплит
RESULTS_DIR  = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Сколько примеров использовать (уменьши для быстрого теста, напр. 50/20)
N_TRAIN = 2000   # примеров для обучения LoRA
N_EVAL  = 200    # примеров для оценки (берём из конца датасета)

LORA_CONFIG = dict(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

TRAIN_CONFIG = dict(
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,   # эффективный батч = 16
    learning_rate=2e-4,
    warmup_ratio=0.1,
    logging_steps=20,
    save_strategy="no",
    dataloader_pin_memory=False,
    bf16=False,   # MPS не поддерживает bf16 при обучении
    fp16=False,
    report_to="none",
)


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    return ' '.join(s.split())


def compute_em(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def compute_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def aggregate(results: list[dict]) -> dict:
    em = sum(r["em"] for r in results) / len(results)
    f1 = sum(r["f1"] for r in results) / len(results)
    return {"EM": round(em * 100, 2), "F1": round(f1 * 100, 2), "n": len(results)}


# ---------------------------------------------------------------------------
# Промпты
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Ты — помощник для ответов на вопросы на русском языке. "
    "Отвечай кратко (1–5 слов). Не объясняй и не перефразируй — только ответ."
)

SYSTEM_PROMPT_WITH_CTX = (
    "Ты — помощник для ответов на вопросы на русском языке. "
    "Отвечай кратко (1–5 слов), используя ТОЛЬКО информацию из контекста. "
    "Не объясняй, не перефразируй — только ответ."
)


# Qwen3 поддерживает thinking-режим; отключаем его добавляя пустой <think> блок.
# Это стандартный способ — модель видит что "думать" уже нечего и сразу отвечает.
_NO_THINK = "<think>\n</think>\n"


def make_prompt_no_context(question: str) -> str:
    """Вариант 1: только вопрос, без параграфа."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nВопрос: {question}<|im_end|>\n"
        f"<|im_start|>assistant\n{_NO_THINK}"
    )


def make_prompt_with_context(context: str, question: str) -> str:
    """Вариант 2 и 3: вопрос + параграф."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT_WITH_CTX}<|im_end|>\n"
        f"<|im_start|>user\nКонтекст: {context}\n\nВопрос: {question}<|im_end|>\n"
        f"<|im_start|>assistant\n{_NO_THINK}"
    )


def make_training_example(example: dict) -> dict:
    """Формат для SFT-обучения LoRA."""
    return {
        "text": (
            f"<|im_start|>system\n{SYSTEM_PROMPT_WITH_CTX}<|im_end|>\n"
            f"<|im_start|>user\nКонтекст: {example['C']}\n\nВопрос: {example['Q']}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['A']}<|im_end|>"
        )
    }


# ---------------------------------------------------------------------------
# Инференс
# ---------------------------------------------------------------------------

def generate(model, tokenizer, prompt: str, device, max_new_tokens: int = 80) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if "</think>" in answer:
        answer = answer.split("</think>", 1)[1].strip()
    return answer


def run_evaluation(model, tokenizer, examples, device, label: str,
                   use_context: bool = True) -> list[dict]:
    model.eval()
    results = []
    for i, ex in enumerate(examples):
        context  = ex["C"]
        question = ex["Q"]
        gold     = ex["A"]

        if use_context:
            prompt = make_prompt_with_context(context, question)
        else:
            prompt = make_prompt_no_context(question)

        pred = generate(model, tokenizer, prompt, device)
        results.append({
            "question": question,
            "context":  context[:200] + "..." if len(context) > 200 else context,
            "gold": gold,
            "pred": pred,
            "em":   compute_em(pred, gold),
            "f1":   compute_f1(pred, gold),
        })

        if (i + 1) % 50 == 0:
            m = aggregate(results)
            print(f"  [{label}] {i+1}/{len(examples)} — EM={m['EM']}, F1={m['F1']}")

    return results


def print_examples(results: list[dict], n: int = 3):
    for r in results[:n]:
        print(f"  Q:    {r['question']}")
        print(f"  Gold: {r['gold']}")
        print(f"  Pred: {r['pred']}")
        print(f"  EM={r['em']:.0f}  F1={r['f1']:.2f}")
        print()


# ---------------------------------------------------------------------------
# Главный скрипт
# ---------------------------------------------------------------------------

def main():
    device = get_device()
    print(f"Устройство: {device}")
    print(f"Модель:     {MODEL_NAME}")

    # -----------------------------------------------------------------------
    # 1. Данные
    # -----------------------------------------------------------------------
    print(f"\n[1] Загрузка {DATASET_NAME}...")
    full = load_dataset(DATASET_NAME)["train"]
    # Конец датасета — eval, начало — train (без пересечений)
    eval_data  = full.select(range(len(full) - N_EVAL, len(full)))
    train_data = full.select(range(N_TRAIN))
    print(f"  Всего: {len(full)} | Обучение: {N_TRAIN} | Оценка: {N_EVAL}")
    ex0 = full[0]
    print(f"  Пример — Q: {ex0['Q']}")
    print(f"           A: {ex0['A']}")

    # -----------------------------------------------------------------------
    # 2. Загрузка модели
    # -----------------------------------------------------------------------
    print(f"\n[2] Загрузка {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,  # float32 для стабильного обучения на MPS
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Параметры: {n_params/1e6:.1f}M")

    # -----------------------------------------------------------------------
    # 3. Вариант 1 — Без контекста
    # -----------------------------------------------------------------------
    print(f"\n[3] Вариант 1: без контекста ({N_EVAL} примеров)...")
    results_no_ctx = run_evaluation(
        model, tokenizer, eval_data, device,
        label="no_context", use_context=False,
    )
    metrics_no_ctx = aggregate(results_no_ctx)
    print(f"  Результат: EM={metrics_no_ctx['EM']}, F1={metrics_no_ctx['F1']}")
    print("\n  Примеры:")
    print_examples(results_no_ctx)

    # -----------------------------------------------------------------------
    # 4. Вариант 2 — С контекстом (без LoRA)
    # -----------------------------------------------------------------------
    print(f"\n[4] Вариант 2: с контекстом, без LoRA ({N_EVAL} примеров)...")
    results_with_ctx = run_evaluation(
        model, tokenizer, eval_data, device,
        label="with_context", use_context=True,
    )
    metrics_with_ctx = aggregate(results_with_ctx)
    print(f"  Результат: EM={metrics_with_ctx['EM']}, F1={metrics_with_ctx['F1']}")
    print("\n  Примеры:")
    print_examples(results_with_ctx)

    # -----------------------------------------------------------------------
    # 5. Добавляем LoRA и обучаем
    # -----------------------------------------------------------------------
    print("\n[5] Добавление LoRA-адаптера...")
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Обучаемые параметры: {trainable:,} ({100*trainable/total:.2f}%)")

    print(f"\n[6] Обучение LoRA на {N_TRAIN} примерах...")
    train_formatted = train_data.map(make_training_example, remove_columns=train_data.column_names)

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(RESULTS_DIR / "lora_russian_qa"),
            **TRAIN_CONFIG,
        ),
        train_dataset=train_formatted,
        processing_class=tokenizer,
    )
    trainer.train()

    lora_path = RESULTS_DIR / "lora_russian_qa" / "adapter"
    model.save_pretrained(str(lora_path))
    print(f"  LoRA сохранён: {lora_path}")

    # -----------------------------------------------------------------------
    # 6. Вариант 3 — LoRA + контекст
    # -----------------------------------------------------------------------
    print(f"\n[7] Вариант 3: LoRA + контекст ({N_EVAL} примеров)...")
    results_lora = run_evaluation(
        model, tokenizer, eval_data, device,
        label="lora", use_context=True,
    )
    metrics_lora = aggregate(results_lora)
    print(f"  Результат: EM={metrics_lora['EM']}, F1={metrics_lora['F1']}")
    print("\n  Примеры:")
    print_examples(results_lora)

    # -----------------------------------------------------------------------
    # 7. Итоговая таблица
    # -----------------------------------------------------------------------
    print("\n" + "=" * 58)
    print("ИТОГИ: Qwen3-0.6B на русском QA (SberSQuAD)")
    print("=" * 58)
    print(f"{'Метод':<30} {'EM':>8} {'F1':>8}")
    print("-" * 58)
    rows = [
        ("(1) Без контекста",          metrics_no_ctx),
        ("(2) С контекстом (no LoRA)", metrics_with_ctx),
        ("(3) LoRA + контекст",        metrics_lora),
    ]
    for name, m in rows:
        print(f"  {name:<28} {m['EM']:>8.2f} {m['F1']:>8.2f}")
    print("=" * 58)

    # -----------------------------------------------------------------------
    # 8. Сохранение
    # -----------------------------------------------------------------------
    output = {
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "n_train": N_TRAIN,
        "n_eval": N_EVAL,
        "no_context":   {"metrics": metrics_no_ctx,   "examples": results_no_ctx[:5]},
        "with_context": {"metrics": metrics_with_ctx, "examples": results_with_ctx[:5]},
        "lora":         {"metrics": metrics_lora,     "examples": results_lora[:5]},
    }
    out_path = RESULTS_DIR / "russian_qa.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nРезультаты сохранены: {out_path}")


if __name__ == "__main__":
    main()
