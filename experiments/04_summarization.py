"""
Эксперимент 4: Summarization — английский (SAMSum)
===================================================
Сравниваем три варианта Qwen3-0.6B:
  (1) Zero-shot  — просим суммаризировать без примеров
  (2) Few-shot   — 2 примера диалог→резюме в промпте
  (3) LoRA       — дообученная модель

Датасет: spencer/samsum_reformat (dialogue, summary)
Метрика: ROUGE-1, ROUGE-2, ROUGE-L

Запуск:
    uv run experiments/04_summarization.py
"""

import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

MODEL_NAME   = "Qwen/Qwen3-0.6B"
DATASET_NAME = "spencer/samsum_reformat"
RESULTS_DIR  = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_TRAIN    = 2000
N_EVAL     = 200
N_FEW_SHOT = 2    # SAMSum диалоги длинные — 2 примера достаточно

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
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # диалоги длиннее — меньше батч
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
# Утилиты
# ---------------------------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def compute_rouge(pred: str, gold: str) -> dict:
    scores = _scorer.score(gold, pred)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def aggregate(results: list[dict]) -> dict:
    r1 = sum(r["rouge1"] for r in results) / len(results)
    r2 = sum(r["rouge2"] for r in results) / len(results)
    rl = sum(r["rougeL"] for r in results) / len(results)
    return {
        "ROUGE-1": round(r1 * 100, 2),
        "ROUGE-2": round(r2 * 100, 2),
        "ROUGE-L": round(rl * 100, 2),
        "n": len(results),
    }


# ---------------------------------------------------------------------------
# Промпты
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a summarization assistant. "
    "Write a concise one-sentence summary of the dialogue. "
    "Be brief and factual."
)

_NO_THINK = "<think>\n</think>\n"


def make_prompt_zero_shot(dialogue: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nDialogue:\n{dialogue}<|im_end|>\n"
        f"<|im_start|>assistant\n{_NO_THINK}"
    )


def make_prompt_few_shot(dialogue: str, examples: list[dict]) -> str:
    shots = ""
    for ex in examples:
        shots += (
            f"<|im_start|>user\nDialogue:\n{ex['dialogue']}<|im_end|>\n"
            f"<|im_start|>assistant\n{ex['summary']}<|im_end|>\n"
        )
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"{shots}"
        f"<|im_start|>user\nDialogue:\n{dialogue}<|im_end|>\n"
        f"<|im_start|>assistant\n{_NO_THINK}"
    )


def make_training_example(example: dict) -> dict:
    return {
        "text": (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nDialogue:\n{example['dialogue']}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['summary']}<|im_end|>"
        )
    }


# ---------------------------------------------------------------------------
# Инференс
# ---------------------------------------------------------------------------

def generate(model, tokenizer, prompt: str, device, max_new_tokens: int = 120) -> str:
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


def run_evaluation(model, tokenizer, examples, few_shot_examples,
                   device, label: str, mode: str = "zero_shot") -> list[dict]:
    model.eval()
    results = []
    for i, ex in enumerate(examples):
        dialogue = ex["dialogue"]
        gold     = ex["summary"]

        prompt = make_prompt_zero_shot(dialogue) if mode == "zero_shot" \
            else make_prompt_few_shot(dialogue, few_shot_examples)

        pred    = generate(model, tokenizer, prompt, device)
        metrics = compute_rouge(pred, gold)
        results.append({"dialogue": dialogue[:150] + "...", "gold": gold,
                         "pred": pred, **metrics})

        if (i + 1) % 50 == 0:
            m = aggregate(results)
            print(f"  [{label}] {i+1}/{len(examples)} — R1={m['ROUGE-1']}, R2={m['ROUGE-2']}, RL={m['ROUGE-L']}")

    return results


def print_examples(results: list[dict], n: int = 2):
    for r in results[:n]:
        print(f"  Dialogue: {r['dialogue']}")
        print(f"  Gold:     {r['gold']}")
        print(f"  Pred:     {r['pred']}")
        print(f"  R1={r['rouge1']:.2f}  R2={r['rouge2']:.2f}  RL={r['rougeL']:.2f}")
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
    ds         = load_dataset(DATASET_NAME)
    train_data = ds["train"].select(range(N_TRAIN))
    eval_split = "validation" if "validation" in ds else "test"
    eval_data  = ds[eval_split].select(range(N_EVAL))
    few_shot   = ds["train"].select(range(N_TRAIN, N_TRAIN + N_FEW_SHOT))
    print(f"  Splits: {list(ds.keys())} | Train: {len(train_data)} | Eval: {len(eval_data)}")

    ex0 = ds["train"][0]
    print(f"  Example dialogue: {ex0['dialogue'][:100]}...")
    print(f"  Example summary:  {ex0['summary']}")

    # -----------------------------------------------------------------------
    # 2. Модель
    # -----------------------------------------------------------------------
    print(f"\n[2] Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # -----------------------------------------------------------------------
    # 3. Вариант 1 — Zero-shot
    # -----------------------------------------------------------------------
    print(f"\n[3] Variant 1: zero-shot ({N_EVAL} examples)...")
    results_zero = run_evaluation(model, tokenizer, eval_data, few_shot,
                                  device, "zero_shot", mode="zero_shot")
    metrics_zero = aggregate(results_zero)
    print(f"  Result: R1={metrics_zero['ROUGE-1']}, R2={metrics_zero['ROUGE-2']}, RL={metrics_zero['ROUGE-L']}")
    print_examples(results_zero)

    # -----------------------------------------------------------------------
    # 4. Вариант 2 — Few-shot
    # -----------------------------------------------------------------------
    print(f"\n[4] Variant 2: {N_FEW_SHOT}-shot ({N_EVAL} examples)...")
    results_few = run_evaluation(model, tokenizer, eval_data, few_shot,
                                 device, "few_shot", mode="few_shot")
    metrics_few = aggregate(results_few)
    print(f"  Result: R1={metrics_few['ROUGE-1']}, R2={metrics_few['ROUGE-2']}, RL={metrics_few['ROUGE-L']}")
    print_examples(results_few)

    # -----------------------------------------------------------------------
    # 5–6. LoRA
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
            output_dir=str(RESULTS_DIR / "lora_summarization"),
            **TRAIN_CONFIG,
        ),
        train_dataset=train_formatted,
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(str(RESULTS_DIR / "lora_summarization" / "adapter"))

    # -----------------------------------------------------------------------
    # 7. Вариант 3 — LoRA
    # -----------------------------------------------------------------------
    print(f"\n[7] Variant 3: LoRA ({N_EVAL} examples)...")
    results_lora = run_evaluation(model, tokenizer, eval_data, few_shot,
                                  device, "lora", mode="zero_shot")
    metrics_lora = aggregate(results_lora)
    print(f"  Result: R1={metrics_lora['ROUGE-1']}, R2={metrics_lora['ROUGE-2']}, RL={metrics_lora['ROUGE-L']}")
    print_examples(results_lora)

    # -----------------------------------------------------------------------
    # 8. Итоговая таблица
    # -----------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("RESULTS: Qwen3-0.6B on Summarization (SAMSum)")
    print("=" * 62)
    print(f"  {'Method':<30} {'R-1':>8} {'R-2':>8} {'R-L':>8}")
    print("-" * 62)
    for name, m in [
        ("(1) Zero-shot",            metrics_zero),
        ("(2) Few-shot (2 examples)", metrics_few),
        ("(3) LoRA",                 metrics_lora),
    ]:
        print(f"  {name:<30} {m['ROUGE-1']:>8.2f} {m['ROUGE-2']:>8.2f} {m['ROUGE-L']:>8.2f}")
    print("=" * 62)

    output = {
        "model": MODEL_NAME, "dataset": DATASET_NAME, "task": "Summarization",
        "n_train": N_TRAIN, "n_eval": N_EVAL,
        "zero_shot": {"metrics": metrics_zero, "examples": results_zero[:5]},
        "few_shot":  {"metrics": metrics_few,  "examples": results_few[:5]},
        "lora":      {"metrics": metrics_lora, "examples": results_lora[:5]},
    }
    out_path = RESULTS_DIR / "summarization.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
