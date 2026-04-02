"""
Эксперимент 3: NER (Named Entity Recognition) — английский
===========================================================
Сравниваем три варианта Qwen3-0.6B:
  (1) Без примеров   — просто просим найти сущности
  (2) С примерами    — 3 few-shot примера в промпте
  (3) LoRA           — дообученная модель

Датасет: eriktks/conll2003
Метрика: F1 по токенам (entity-level)

Запуск:
    uv run experiments/03_ner.py
"""

import json
import re
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
DATASET_NAME = "Davlan/conll2003_noMISC"
RESULTS_DIR  = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_TRAIN      = 2000
N_EVAL       = 200
N_FEW_SHOT   = 3     # примеров в промпте для варианта 2

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

# Davlan/conll2003_noMISC: метки уже строки, MISC отсутствует
NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

# ---------------------------------------------------------------------------
# Утилиты датасета
# ---------------------------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def decode_ner(tokens: list[str], tag_ids: list) -> dict[str, list[str]]:
    """Извлечь именованные сущности из BIO-разметки.
    tag_ids может быть списком строк ('B-ORG') или чисел (индексы NER_LABELS).
    """
    entities = {"PER": [], "ORG": [], "LOC": []}
    current, current_type = [], None
    for token, tag_id in zip(tokens, tag_ids):
        label = tag_id if isinstance(tag_id, str) else (NER_LABELS[tag_id] if tag_id < len(NER_LABELS) else "O")
        if label.startswith("B-"):
            if current:
                entities[current_type].append(" ".join(current))
            current = [token]
            current_type = label[2:]
        elif label.startswith("I-") and current_type == label[2:]:
            current.append(token)
        else:
            if current:
                entities[current_type].append(" ".join(current))
            current, current_type = [], None
    if current:
        entities[current_type].append(" ".join(current))
    return entities


def entities_to_string(entities: dict[str, list[str]]) -> str:
    """Сущности в строку для промпта и сравнения."""
    parts = []
    for etype, ents in entities.items():
        if ents:
            parts.append(f"{etype}: {', '.join(ents)}")
    return " | ".join(parts) if parts else "none"


def compute_ner_f1(pred_str: str, gold_entities: dict[str, list[str]]) -> dict:
    """F1 на уровне сущностей."""
    def parse_pred(s: str) -> set[str]:
        found = set()
        for part in s.split("|"):
            part = part.strip()
            if ":" in part:
                etype, rest = part.split(":", 1)
                etype = etype.strip().upper()
                for ent in rest.split(","):
                    ent = ent.strip().lower()
                    if ent and ent != "none":
                        found.add(f"{etype}:{ent}")
        return found

    gold_set = set()
    for etype, ents in gold_entities.items():
        for ent in ents:
            gold_set.add(f"{etype}:{ent.lower()}")

    pred_set = parse_pred(pred_str.lower())

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"f1": f1, "precision": precision, "recall": recall}


def aggregate(results: list[dict]) -> dict:
    f1   = sum(r["f1"]        for r in results) / len(results)
    prec = sum(r["precision"] for r in results) / len(results)
    rec  = sum(r["recall"]    for r in results) / len(results)
    return {
        "F1":        round(f1   * 100, 2),
        "Precision": round(prec * 100, 2),
        "Recall":    round(rec  * 100, 2),
        "n": len(results),
    }


# ---------------------------------------------------------------------------
# Промпты
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a Named Entity Recognition (NER) assistant. "
    "Find all named entities in the sentence and list them by type. "
    "Types: PER (person), ORG (organization), LOC (location). "
    "Format: 'PER: name1, name2 | ORG: org1 | LOC: loc1'. "
    "If no entities of a type, skip it. If no entities at all, write 'none'. "
    "Do NOT explain your reasoning. Do NOT think step by step. "
    "Output ONLY the entity list in the exact format above, nothing else."
)

_NO_THINK = "<think>\n</think>\n"


def make_prompt_zero_shot(sentence: str, tokenizer=None) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nSentence: {sentence}<|im_end|>\n"
        f"<|im_start|>assistant\n{_NO_THINK}"
    )


def make_prompt_few_shot(sentence: str, examples: list[dict], tokenizer=None) -> str:
    shots = ""
    for ex in examples:
        gold_str = entities_to_string(decode_ner(ex["tokens"], ex["ner_tags"]))
        shots += (
            f"<|im_start|>user\nSentence: {' '.join(ex['tokens'])}<|im_end|>\n"
            f"<|im_start|>assistant\n{gold_str}<|im_end|>\n"
        )
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"{shots}"
        f"<|im_start|>user\nSentence: {sentence}<|im_end|>\n"
        f"<|im_start|>assistant\n{_NO_THINK}"
    )


def make_training_example(example: dict) -> dict:
    sentence  = " ".join(example["tokens"])
    gold_str  = entities_to_string(decode_ner(example["tokens"], example["ner_tags"]))
    return {
        "text": (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nSentence: {sentence}<|im_end|>\n"
            f"<|im_start|>assistant\n{gold_str}<|im_end|>"
        )
    }


# ---------------------------------------------------------------------------
# Инференс
# ---------------------------------------------------------------------------

def extract_ner_answer(text: str) -> str:
    """Пост-обработка: извлечь NER-ответ из сырого вывода модели.

    Если модель начала рассуждать вместо того чтобы дать ответ в формате
    'PER: ... | ORG: ... | LOC: ...', пытаемся найти ответ по паттернам.
    """
    # Убираем всё до </think> если модель всё же вошла в thinking
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    # Если ответ уже в правильном формате — возвращаем как есть
    # Паттерн: начинается с PER:/ORG:/LOC: или равен "none"
    first_line = text.split("\n")[0].strip()
    if re.match(r"^(PER|ORG|LOC)\s*:", first_line, re.IGNORECASE) or first_line.lower() == "none":
        return first_line

    # Ищем строку в формате "TYPE: entity" где-то внутри рассуждений
    ner_pattern = re.compile(
        r"((?:PER|ORG|LOC)\s*:\s*[^|\n]+(?:\s*\|\s*(?:PER|ORG|LOC)\s*:\s*[^|\n]+)*)",
        re.IGNORECASE,
    )
    matches = ner_pattern.findall(text)
    if matches:
        # Берём последнее совпадение — обычно это финальный ответ
        return matches[-1].strip()

    # Если ничего не нашли — "none" безопаснее чем мусор
    return "none"


def generate(model, tokenizer, prompt: str, device, max_new_tokens: int = 100) -> str:
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
    return extract_ner_answer(text)


def run_evaluation(model, tokenizer, examples, few_shot_examples,
                   device, label: str, mode: str = "zero_shot") -> list[dict]:
    model.eval()
    results = []
    for i, ex in enumerate(examples):
        sentence      = " ".join(ex["tokens"])
        gold_entities = decode_ner(ex["tokens"], ex["ner_tags"])

        if mode == "zero_shot":
            prompt = make_prompt_zero_shot(sentence, tokenizer=tokenizer)
        else:
            prompt = make_prompt_few_shot(sentence, few_shot_examples, tokenizer=tokenizer)

        pred = generate(model, tokenizer, prompt, device)
        metrics = compute_ner_f1(pred, gold_entities)
        results.append({
            "sentence":      sentence,
            "gold":          entities_to_string(gold_entities),
            "pred":          pred,
            **metrics,
        })

        if (i + 1) % 50 == 0:
            m = aggregate(results)
            print(f"  [{label}] {i+1}/{len(examples)} — F1={m['F1']}")

    return results


def print_examples(results: list[dict], n: int = 3):
    for r in results[:n]:
        print(f"  Sent: {r['sentence']}")
        print(f"  Gold: {r['gold']}")
        print(f"  Pred: {r['pred']}")
        print(f"  F1={r['f1']:.2f}")
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
    eval_data  = ds["validation"].select(range(N_EVAL))
    few_shot   = ds["train"].select(range(N_TRAIN, N_TRAIN + N_FEW_SHOT))
    print(f"  Train: {len(train_data)} | Eval: {len(eval_data)}")

    ex0 = ds["train"][0]
    print(f"  Example tokens: {ex0['tokens'][:8]}")
    print(f"  Example labels: {ex0['ner_tags'][:8]}")

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
    # 3. Вариант 1 — Zero-shot (без примеров)
    # -----------------------------------------------------------------------
    print(f"\n[3] Variant 1: zero-shot ({N_EVAL} examples)...")
    results_zero = run_evaluation(model, tokenizer, eval_data, few_shot,
                                  device, "zero_shot", mode="zero_shot")
    metrics_zero = aggregate(results_zero)
    print(f"  Result: F1={metrics_zero['F1']}, P={metrics_zero['Precision']}, R={metrics_zero['Recall']}")
    print_examples(results_zero)

    # -----------------------------------------------------------------------
    # 4. Вариант 2 — Few-shot (с примерами в промпте)
    # -----------------------------------------------------------------------
    print(f"\n[4] Variant 2: {N_FEW_SHOT}-shot ({N_EVAL} examples)...")
    results_few = run_evaluation(model, tokenizer, eval_data, few_shot,
                                 device, "few_shot", mode="few_shot")
    metrics_few = aggregate(results_few)
    print(f"  Result: F1={metrics_few['F1']}, P={metrics_few['Precision']}, R={metrics_few['Recall']}")
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
            output_dir=str(RESULTS_DIR / "lora_ner"),
            **TRAIN_CONFIG,
        ),
        train_dataset=train_formatted,
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(str(RESULTS_DIR / "lora_ner" / "adapter"))

    # -----------------------------------------------------------------------
    # 7. Вариант 3 — LoRA
    # -----------------------------------------------------------------------
    print(f"\n[7] Variant 3: LoRA ({N_EVAL} examples)...")
    results_lora = run_evaluation(model, tokenizer, eval_data, few_shot,
                                  device, "lora", mode="zero_shot")
    metrics_lora = aggregate(results_lora)
    print(f"  Result: F1={metrics_lora['F1']}, P={metrics_lora['Precision']}, R={metrics_lora['Recall']}")
    print_examples(results_lora)

    # -----------------------------------------------------------------------
    # 8. Итоговая таблица
    # -----------------------------------------------------------------------
    print("\n" + "=" * 58)
    print("RESULTS: Qwen3-0.6B on NER (CoNLL2003)")
    print("=" * 58)
    print(f"  {'Method':<30} {'F1':>8} {'P':>8} {'R':>8}")
    print("-" * 58)
    for name, m in [
        ("(1) Zero-shot",          metrics_zero),
        ("(2) Few-shot (3 examples)", metrics_few),
        ("(3) LoRA",               metrics_lora),
    ]:
        print(f"  {name:<30} {m['F1']:>8.2f} {m['Precision']:>8.2f} {m['Recall']:>8.2f}")
    print("=" * 58)

    output = {
        "model": MODEL_NAME, "dataset": DATASET_NAME, "task": "NER",
        "n_train": N_TRAIN, "n_eval": N_EVAL,
        "zero_shot": {"metrics": metrics_zero, "examples": results_zero[:5]},
        "few_shot":  {"metrics": metrics_few,  "examples": results_few[:5]},
        "lora":      {"metrics": metrics_lora, "examples": results_lora[:5]},
    }
    out_path = RESULTS_DIR / "ner.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
    