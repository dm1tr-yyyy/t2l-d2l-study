"""
Эксперимент 2: Английский язык — QA (SQuAD)
============================================
Сравниваем три варианта Qwen3-0.6B:
  (1) Без контекста  — только вопрос
  (2) С контекстом   — вопрос + параграф в промпте
  (3) LoRA           — дообученная модель (с параграфом)

Используем тот же подход что в 01_russian_qa.py — сравниваем результаты
двух языков в итоговой таблице.

Запуск:
    uv run experiments/02_english_qa.py

Настрой DATASET_NAME и колонки под найденный датасет (см. DATASET CONFIG ниже).
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
# Конфигурация модели
# ---------------------------------------------------------------------------

MODEL_NAME  = "Qwen/Qwen3-0.6B"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_TRAIN = 2000
N_EVAL  = 200

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
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    logging_steps=20,
    save_strategy="no",
    dataloader_pin_memory=False,
    bf16=False,
    fp16=False,
    report_to="none",
)

# ---------------------------------------------------------------------------
# DATASET CONFIG — настрой под свой датасет
# ---------------------------------------------------------------------------

DATASET_NAME = "rajpurkar/squad"

# Имена колонок в датасете (поменяй если отличаются)
COL_CONTEXT  = "context"      # параграф с ответом
COL_QUESTION = "question"     # вопрос
COL_ANSWER   = "answers"      # ответ — см. функцию get_answer() ниже

# Имена сплитов
SPLIT_TRAIN = "train"
SPLIT_EVAL  = "validation"    # может быть "test" или "dev"


def get_answer(example: dict) -> str:
    """Извлечь строку-ответ из примера.

    SQuAD хранит ответы как {"text": [...], "answer_start": [...]}.
    Если у твоего датасета другой формат — поменяй здесь.
    """
    ans = example[COL_ANSWER]
    if isinstance(ans, dict):
        return ans["text"][0]       # SQuAD-формат
    if isinstance(ans, list):
        return ans[0]               # просто список строк
    return str(ans)                 # строка напрямую


# ---------------------------------------------------------------------------
# Утилиты (те же что в русском эксперименте)
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
# Промпты (английские)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a question answering assistant. "
    "Answer briefly (1-5 words). Do not explain — just the answer."
)

SYSTEM_PROMPT_WITH_CTX = (
    "You are a question answering assistant. "
    "Answer briefly (1-5 words) using ONLY information from the context. "
    "Do not explain — just the answer."
)

_NO_THINK = "<think>\n</think>\n"


def make_prompt_no_context(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nQuestion: {question}<|im_end|>\n"
        f"<|im_start|>assistant\n{_NO_THINK}"
    )


def make_prompt_with_context(context: str, question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT_WITH_CTX}<|im_end|>\n"
        f"<|im_start|>user\nContext: {context}\n\nQuestion: {question}<|im_end|>\n"
        f"<|im_start|>assistant\n{_NO_THINK}"
    )


def make_training_example(example: dict) -> dict:
    return {
        "text": (
            f"<|im_start|>system\n{SYSTEM_PROMPT_WITH_CTX}<|im_end|>\n"
            f"<|im_start|>user\nContext: {example[COL_CONTEXT]}\n\n"
            f"Question: {example[COL_QUESTION]}<|im_end|>\n"
            f"<|im_start|>assistant\n{get_answer(example)}<|im_end|>"
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
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


def run_evaluation(model, tokenizer, examples, device, label: str,
                   use_context: bool = True) -> list[dict]:
    model.eval()
    results = []
    for i, ex in enumerate(examples):
        context  = ex[COL_CONTEXT]
        question = ex[COL_QUESTION]
        gold     = get_answer(ex)

        prompt = make_prompt_with_context(context, question) if use_context \
            else make_prompt_no_context(question)

        pred = generate(model, tokenizer, prompt, device)
        results.append({
            "question": question,
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
    print(f"Device:  {device}")
    print(f"Model:   {MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME}")

    # -----------------------------------------------------------------------
    # 1. Данные
    # -----------------------------------------------------------------------
    print(f"\n[1] Loading {DATASET_NAME}...")
    ds = load_dataset(DATASET_NAME)
    print(f"  Splits: {list(ds.keys())}")
    print(f"  Columns: {ds[SPLIT_TRAIN].column_names}")

    train_data = ds[SPLIT_TRAIN].select(range(N_TRAIN))
    eval_split = SPLIT_EVAL if SPLIT_EVAL in ds else list(ds.keys())[-1]
    eval_data  = ds[eval_split].select(range(N_EVAL))
    print(f"  Train: {len(train_data)} | Eval: {len(eval_data)} (from '{eval_split}')")

    ex0 = ds[SPLIT_TRAIN][0]
    print(f"  Example — Q: {ex0[COL_QUESTION]}")
    print(f"             A: {get_answer(ex0)}")

    # -----------------------------------------------------------------------
    # 2. Модель
    # -----------------------------------------------------------------------
    print(f"\n[2] Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    ).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # -----------------------------------------------------------------------
    # 3. Вариант 1 — Без контекста
    # -----------------------------------------------------------------------
    print(f"\n[3] Variant 1: no context ({N_EVAL} examples)...")
    results_no_ctx = run_evaluation(model, tokenizer, eval_data, device,
                                    "no_context", use_context=False)
    metrics_no_ctx = aggregate(results_no_ctx)
    print(f"  Result: EM={metrics_no_ctx['EM']}, F1={metrics_no_ctx['F1']}")
    print_examples(results_no_ctx)

    # -----------------------------------------------------------------------
    # 4. Вариант 2 — С контекстом (без LoRA)
    # -----------------------------------------------------------------------
    print(f"\n[4] Variant 2: with context, no LoRA ({N_EVAL} examples)...")
    results_with_ctx = run_evaluation(model, tokenizer, eval_data, device,
                                      "with_context", use_context=True)
    metrics_with_ctx = aggregate(results_with_ctx)
    print(f"  Result: EM={metrics_with_ctx['EM']}, F1={metrics_with_ctx['F1']}")
    print_examples(results_with_ctx)

    # -----------------------------------------------------------------------
    # 5–6. Обучение LoRA
    # -----------------------------------------------------------------------
    print("\n[5] Adding LoRA adapter...")
    model = get_peft_model(model, LoraConfig(**LORA_CONFIG))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

    print(f"\n[6] Training LoRA on {N_TRAIN} examples...")
    train_formatted = train_data.map(make_training_example,
                                     remove_columns=train_data.column_names)
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(RESULTS_DIR / "lora_english_qa"),
            **TRAIN_CONFIG,
        ),
        train_dataset=train_formatted,
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(str(RESULTS_DIR / "lora_english_qa" / "adapter"))

    # -----------------------------------------------------------------------
    # 7. Вариант 3 — LoRA + контекст
    # -----------------------------------------------------------------------
    print(f"\n[7] Variant 3: LoRA + context ({N_EVAL} examples)...")
    results_lora = run_evaluation(model, tokenizer, eval_data, device,
                                  "lora", use_context=True)
    metrics_lora = aggregate(results_lora)
    print(f"  Result: EM={metrics_lora['EM']}, F1={metrics_lora['F1']}")
    print_examples(results_lora)

    # -----------------------------------------------------------------------
    # 8. Итоговая таблица
    # -----------------------------------------------------------------------
    print("\n" + "=" * 58)
    print(f"RESULTS: Qwen3-0.6B on English QA ({DATASET_NAME})")
    print("=" * 58)
    print(f"  {'Method':<30} {'EM':>8} {'F1':>8}")
    print("-" * 58)
    for name, m in [
        ("(1) No context",           metrics_no_ctx),
        ("(2) With context (no LoRA)", metrics_with_ctx),
        ("(3) LoRA + context",        metrics_lora),
    ]:
        print(f"  {name:<30} {m['EM']:>8.2f} {m['F1']:>8.2f}")
    print("=" * 58)

    # -----------------------------------------------------------------------
    # 9. Сохранение
    # -----------------------------------------------------------------------
    output = {
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "language": "en",
        "n_train": N_TRAIN,
        "n_eval": N_EVAL,
        "no_context":   {"metrics": metrics_no_ctx,   "examples": results_no_ctx[:5]},
        "with_context": {"metrics": metrics_with_ctx, "examples": results_with_ctx[:5]},
        "lora":         {"metrics": metrics_lora,     "examples": results_lora[:5]},
    }
    out_path = RESULTS_DIR / "english_qa.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
