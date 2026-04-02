"""
Эксперимент 5: Сравнение времени инференса по задачам и вариантам
=================================================================
Меряем время на 200 примерах для каждого из 3 вариантов:
  (1) Без контекста / zero-shot
  (2) С контекстом / few-shot
  (3) LoRA (загружаем сохранённый адаптер)

Задачи: Russian QA, English QA, NER, Summarization

Запуск:
    uv run experiments/05_inference_time.py

Результаты: experiments/results/inference_time.json
"""

import json
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

MODEL_NAME  = "Qwen/Qwen3-0.6B"
RESULTS_DIR = Path(__file__).parent / "results"
N_EVAL      = 200
_NO_THINK   = "<think>\n</think>\n"

# Пути к сохранённым адаптерам
ADAPTERS = {
    "russian_qa":      RESULTS_DIR / "lora_russian_qa"  / "adapter",
    "english_qa":      RESULTS_DIR / "lora_english_qa"  / "adapter",
    "summarization":   RESULTS_DIR / "lora_summarization" / "adapter",
    "ner":             RESULTS_DIR / "lora_ner" / "adapter",
}

# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def generate(model, tokenizer, prompt: str, device, max_new_tokens: int = 80) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


def measure_time(model, tokenizer, prompts: list[str], device,
                 max_new_tokens: int = 80, label: str = "") -> dict:
    """Прогоняем все промпты и меряем суммарное и среднее время."""
    model.eval()
    times = []
    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        generate(model, tokenizer, prompt, device, max_new_tokens)
        times.append(time.perf_counter() - t0)
        if (i + 1) % 50 == 0:
            print(f"  [{label}] {i+1}/{len(prompts)} — avg {sum(times)/len(times):.2f}s/example")

    total   = sum(times)
    per_ex  = total / len(times)
    return {
        "total_sec":      round(total, 2),
        "per_example_sec": round(per_ex, 3),
        "n": len(times),
    }


# ---------------------------------------------------------------------------
# Промпты для каждой задачи
# ---------------------------------------------------------------------------

def russian_qa_prompts(examples, mode: str) -> list[str]:
    SYSTEM = (
        "Ты — помощник для ответов на вопросы на русском языке. "
        "Отвечай кратко (1–5 слов). Не объясняй — только ответ."
    )
    SYSTEM_CTX = (
        "Ты — помощник для ответов на вопросы на русском языке. "
        "Отвечай кратко (1–5 слов), используя ТОЛЬКО информацию из контекста. "
        "Не объясняй — только ответ."
    )
    prompts = []
    for ex in examples:
        if mode == "no_context":
            prompts.append(
                f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\nВопрос: {ex['Q']}<|im_end|>\n"
                f"<|im_start|>assistant\n{_NO_THINK}"
            )
        else:
            prompts.append(
                f"<|im_start|>system\n{SYSTEM_CTX}<|im_end|>\n"
                f"<|im_start|>user\nКонтекст: {ex['C']}\n\nВопрос: {ex['Q']}<|im_end|>\n"
                f"<|im_start|>assistant\n{_NO_THINK}"
            )
    return prompts


def english_qa_prompts(examples, mode: str) -> list[str]:
    SYSTEM = (
        "You are a question answering assistant. "
        "Answer briefly (1-5 words). Do not explain — just the answer."
    )
    SYSTEM_CTX = (
        "You are a question answering assistant. "
        "Answer briefly (1-5 words) using ONLY information from the context. "
        "Do not explain — just the answer."
    )
    prompts = []
    for ex in examples:
        if mode == "no_context":
            prompts.append(
                f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\nQuestion: {ex['question']}<|im_end|>\n"
                f"<|im_start|>assistant\n{_NO_THINK}"
            )
        else:
            prompts.append(
                f"<|im_start|>system\n{SYSTEM_CTX}<|im_end|>\n"
                f"<|im_start|>user\nContext: {ex['context']}\n\nQuestion: {ex['question']}<|im_end|>\n"
                f"<|im_start|>assistant\n{_NO_THINK}"
            )
    return prompts


def ner_prompts(examples, mode: str, few_shot_examples=None) -> list[str]:
    SYSTEM = (
        "You are a Named Entity Recognition (NER) assistant. "
        "Find all named entities in the sentence and list them by type. "
        "Types: PER (person), ORG (organization), LOC (location). "
        "Format: 'PER: name1 | ORG: org1 | LOC: loc1'. "
        "If no entities, write 'none'."
    )
    prompts = []
    for ex in examples:
        sentence = " ".join(ex["tokens"])
        if mode == "zero_shot":
            prompts.append(
                f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\nSentence: {sentence}<|im_end|>\n"
                f"<|im_start|>assistant\n{_NO_THINK}"
            )
        else:
            shots = ""
            for fex in (few_shot_examples or []):
                fsent = " ".join(fex["tokens"])
                shots += (
                    f"<|im_start|>user\nSentence: {fsent}<|im_end|>\n"
                    f"<|im_start|>assistant\nnone<|im_end|>\n"
                )
            prompts.append(
                f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
                f"{shots}"
                f"<|im_start|>user\nSentence: {sentence}<|im_end|>\n"
                f"<|im_start|>assistant\n{_NO_THINK}"
            )
    return prompts


def summarization_prompts(examples, mode: str, few_shot_examples=None) -> list[str]:
    SYSTEM = (
        "You are a summarization assistant. "
        "Write a concise one-sentence summary of the dialogue. "
        "Be brief and factual."
    )
    prompts = []
    for ex in examples:
        if mode == "zero_shot":
            prompts.append(
                f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\nDialogue:\n{ex['dialogue']}<|im_end|>\n"
                f"<|im_start|>assistant\n{_NO_THINK}"
            )
        else:
            shots = ""
            for fex in (few_shot_examples or []):
                shots += (
                    f"<|im_start|>user\nDialogue:\n{fex['dialogue']}<|im_end|>\n"
                    f"<|im_start|>assistant\n{fex['summary']}<|im_end|>\n"
                )
            prompts.append(
                f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
                f"{shots}"
                f"<|im_start|>user\nDialogue:\n{ex['dialogue']}<|im_end|>\n"
                f"<|im_start|>assistant\n{_NO_THINK}"
            )
    return prompts


# ---------------------------------------------------------------------------
# Главный скрипт
# ---------------------------------------------------------------------------

def run_task(task_name, base_model, tokenizer, device,
             prompts_v1, prompts_v2, prompts_v3,
             adapter_path, max_new_tokens=80):
    """Запускает три варианта для одной задачи, возвращает результаты."""
    print(f"\n{'='*55}")
    print(f"Задача: {task_name}")
    print(f"{'='*55}")
    results = {}

    # Вариант 1
    print(f"\n  Вариант 1 ({len(prompts_v1)} примеров)...")
    results["v1"] = measure_time(base_model, tokenizer, prompts_v1, device,
                                  max_new_tokens, label="v1")
    print(f"  → {results['v1']['total_sec']:.1f}s total | {results['v1']['per_example_sec']:.2f}s/example")

    # Вариант 2
    print(f"\n  Вариант 2 ({len(prompts_v2)} примеров)...")
    results["v2"] = measure_time(base_model, tokenizer, prompts_v2, device,
                                  max_new_tokens, label="v2")
    print(f"  → {results['v2']['total_sec']:.1f}s total | {results['v2']['per_example_sec']:.2f}s/example")

    # Вариант 3 — LoRA
    if adapter_path and adapter_path.exists():
        print(f"\n  Вариант 3 — LoRA ({len(prompts_v3)} примеров)...")
        lora_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        results["v3"] = measure_time(lora_model, tokenizer, prompts_v3, device,
                                      max_new_tokens, label="v3_lora")
        print(f"  → {results['v3']['total_sec']:.1f}s total | {results['v3']['per_example_sec']:.2f}s/example")
        # unload() возвращает чистую базовую модель без peft_config
        base_model = lora_model.unload()
    else:
        print(f"\n  Вариант 3 — LoRA: адаптер не найден, пропускаем")
        results["v3"] = None

    return results, base_model


def print_summary(all_results: dict):
    v_names = {
        "russian_qa":    ("(1) Без контекста", "(2) С контекстом", "(3) LoRA"),
        "english_qa":    ("(1) No context",    "(2) With context", "(3) LoRA"),
        "ner":           ("(1) Zero-shot",      "(2) Few-shot",     "(3) LoRA"),
        "summarization": ("(1) Zero-shot",      "(2) Few-shot",     "(3) LoRA"),
    }
    print("\n" + "="*70)
    print("ИТОГИ: время инференса на 200 примерах (сек/пример)")
    print("="*70)
    print(f"  {'Задача':<20} {'Вариант':<28} {'сек/пример':>12} {'всего (с)':>10}")
    print("-"*70)
    for task, res in all_results.items():
        names = v_names.get(task, ("V1", "V2", "V3"))
        for i, (vkey, vname) in enumerate(zip(["v1", "v2", "v3"], names)):
            m = res.get(vkey)
            if m:
                print(f"  {task if i==0 else '':<20} {vname:<28} {m['per_example_sec']:>12.3f} {m['total_sec']:>10.1f}")
            else:
                print(f"  {task if i==0 else '':<20} {vname:<28} {'—':>12} {'—':>10}")
    print("="*70)


def main():
    device = get_device()
    print(f"Устройство: {device}")
    print(f"Модель:     {MODEL_NAME}")
    print(f"Примеров:   {N_EVAL} на задачу")

    # -----------------------------------------------------------------------
    # Загрузка модели (один раз для всех задач)
    # -----------------------------------------------------------------------
    print(f"\nЗагрузка {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float32
    ).to(device)
    base_model.eval()
    print(f"  Параметры: {sum(p.numel() for p in base_model.parameters())/1e6:.1f}M")

    all_results = {}

    # -----------------------------------------------------------------------
    # Задача 1: Russian QA
    # -----------------------------------------------------------------------
    print("\n[Загрузка данных] ERmak1581/QA_sberquad...")
    full = load_dataset("ERmak1581/QA_sberquad")["train"]
    ru_eval = full.select(range(len(full) - N_EVAL, len(full)))

    all_results["russian_qa"], base_model = run_task(
        task_name      = "Russian QA",
        base_model     = base_model,
        tokenizer      = tokenizer,
        device         = device,
        prompts_v1     = russian_qa_prompts(ru_eval, "no_context"),
        prompts_v2     = russian_qa_prompts(ru_eval, "with_context"),
        prompts_v3     = russian_qa_prompts(ru_eval, "with_context"),
        adapter_path   = ADAPTERS["russian_qa"],
        max_new_tokens = 80,
    )

    # -----------------------------------------------------------------------
    # Задача 2: English QA
    # -----------------------------------------------------------------------
    print("\n[Загрузка данных] rajpurkar/squad...")
    squad = load_dataset("rajpurkar/squad")
    en_eval = squad["validation"].select(range(N_EVAL))

    all_results["english_qa"], base_model = run_task(
        task_name      = "English QA",
        base_model     = base_model,
        tokenizer      = tokenizer,
        device         = device,
        prompts_v1     = english_qa_prompts(en_eval, "no_context"),
        prompts_v2     = english_qa_prompts(en_eval, "with_context"),
        prompts_v3     = english_qa_prompts(en_eval, "with_context"),
        adapter_path   = ADAPTERS["english_qa"],
        max_new_tokens = 80,
    )

    # -----------------------------------------------------------------------
    # Задача 3: NER
    # -----------------------------------------------------------------------
    print("\n[Загрузка данных] Davlan/conll2003_noMISC...")
    ner_ds   = load_dataset("Davlan/conll2003_noMISC")
    ner_eval = ner_ds["validation"].select(range(N_EVAL))
    ner_few  = ner_ds["train"].select(range(3))

    all_results["ner"], base_model = run_task(
        task_name      = "NER",
        base_model     = base_model,
        tokenizer      = tokenizer,
        device         = device,
        prompts_v1     = ner_prompts(ner_eval, "zero_shot"),
        prompts_v2     = ner_prompts(ner_eval, "few_shot", ner_few),
        prompts_v3     = ner_prompts(ner_eval, "zero_shot"),
        adapter_path   = ADAPTERS["ner"],
        max_new_tokens = 100,
    )

    # -----------------------------------------------------------------------
    # Задача 4: Summarization
    # -----------------------------------------------------------------------
    print("\n[Загрузка данных] spencer/samsum_reformat...")
    sam_ds   = load_dataset("spencer/samsum_reformat")
    sam_eval = sam_ds["test"].select(range(N_EVAL)) if "test" in sam_ds \
        else sam_ds["validation"].select(range(N_EVAL))
    sam_few  = sam_ds["train"].select(range(2))

    all_results["summarization"], base_model = run_task(
        task_name      = "Summarization",
        base_model     = base_model,
        tokenizer      = tokenizer,
        device         = device,
        prompts_v1     = summarization_prompts(sam_eval, "zero_shot"),
        prompts_v2     = summarization_prompts(sam_eval, "few_shot", sam_few),
        prompts_v3     = summarization_prompts(sam_eval, "zero_shot"),
        adapter_path   = ADAPTERS["summarization"],
        max_new_tokens = 120,
    )

    # -----------------------------------------------------------------------
    # Итоговая таблица
    # -----------------------------------------------------------------------
    print_summary(all_results)

    # Сохраняем
    out_path = RESULTS_DIR / "inference_time.json"
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\nРезультаты сохранены: {out_path}")


if __name__ == "__main__":
    main()
