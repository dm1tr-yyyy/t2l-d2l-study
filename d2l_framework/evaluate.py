"""
Оценка D2L на 200 примерах SQuAD: base → +context → D2L.

Те же метрики что в experiments/02_english_qa.py: EM, F1.
Сводная таблица трёх вариантов + время инференса.

Запуск:
    uv run python -m d2l_framework.evaluate d2l_checkpoints/final.pt
"""

import json
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset

from .config import auto_config
from .inference import DocToLoRAInference

N_EVAL = 50
RESULTS_DIR = Path("d2l_framework/results")


# ---------------------------------------------------------------------------
# Метрики (из experiments/01_russian_qa.py)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    return ' '.join(s.split())


def compute_em(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def compute_f1(pred: str, gold: str) -> float:
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    common = Counter(pt) & Counter(gt)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    p = ns / len(pt)
    r = ns / len(gt)
    return 2 * p * r / (p + r)


def aggregate(results: list[dict]) -> dict:
    em = sum(r["em"] for r in results) / len(results) * 100
    f1 = sum(r["f1"] for r in results) / len(results) * 100
    return {"EM": round(em, 2), "F1": round(f1, 2), "n": len(results)}


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def eval_variant(
    model: DocToLoRAInference,
    examples: list[dict],
    mode: str,
    label: str,
) -> tuple[dict, list[dict], float]:
    """
    mode: 'base' | 'context' | 'd2l'
    Возвращает (metrics, results, sec_per_example)
    """
    results = []
    t_total = 0.0

    for i, ex in enumerate(examples):
        context  = ex["context"]
        question = ex["question"]
        gold     = ex["answers"]["text"][0] if ex["answers"]["text"] else ""

        t0 = time.perf_counter()

        if mode == "base":
            pred = model.generate(question, max_new_tokens=600)

        elif mode == "context":
            pred = model.generate_with_context(context, question, max_new_tokens=600)

        elif mode == "d2l":
            model.internalize(context)
            pred = model.generate(question, max_new_tokens=600)

        t_total += time.perf_counter() - t0

        results.append({
            "question": question,
            "gold": gold,
            "pred": pred,
            "em": compute_em(pred, gold),
            "f1": compute_f1(pred, gold),
        })

        if (i + 1) % 10 == 0:
            m = aggregate(results)
            spe = t_total / len(results)
            print(f"  [{label}] {i+1}/{N_EVAL} — EM={m['EM']:.1f}, F1={m['F1']:.1f} ({spe:.2f}s/ex)")

    metrics = aggregate(results)
    spe = t_total / len(results)
    print(f"  [{label}] Итог: EM={metrics['EM']:.2f}, F1={metrics['F1']:.2f} | {spe:.2f}s/ex")

    # Примеры
    print(f"\n  Примеры ({label}):")
    for r in results[:3]:
        print(f"    Q:    {r['question'][:80]}")
        print(f"    Gold: {r['gold'][:80]}")
        print(f"    Pred: {r['pred'][:80]}")
        print(f"    EM={r['em']:.0f}  F1={r['f1']:.2f}")
        print()

    return metrics, results, spe


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else None

    config = auto_config()
    print("=" * 60)
    print("D2L Evaluation: {N_EVAL} примеров SQuAD (English QA)")
    print(f"Model: {config.model_name}")
    if checkpoint:
        print(f"Checkpoint: {checkpoint}")
    else:
        print("(без чекпоинта — необученная гиперсеть)")
    print("=" * 60)

    # Загружаем данные
    print("\nЗагружаю SQuAD validation...")
    ds = load_dataset("rajpurkar/squad")["validation"]
    examples = list(ds.select(range(N_EVAL)))
    print(f"  {N_EVAL} примеров")

    # Загружаем модель
    print("Загружаю D2L модель...")
    model = DocToLoRAInference(checkpoint_path=checkpoint, config=config)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    # --- Вариант 1: Base ---
    print(f"\n{'='*60}")
    print("ВАРИАНТ 1: Base (без контекста)")
    print(f"{'='*60}")
    m1, r1, t1 = eval_variant(model, examples, "base", "base")
    all_metrics["base"] = {**m1, "sec_per_example": round(t1, 3)}

    # --- Вариант 2: +Context ---
    print(f"\n{'='*60}")
    print("ВАРИАНТ 2: +Context (документ в промпте)")
    print(f"{'='*60}")
    m2, r2, t2 = eval_variant(model, examples, "context", "+context")
    all_metrics["context"] = {**m2, "sec_per_example": round(t2, 3)}

    # --- Вариант 3: D2L ---
    print(f"\n{'='*60}")
    print("ВАРИАНТ 3: D2L (документ → LoRA)")
    print(f"{'='*60}")
    m3, r3, t3 = eval_variant(model, examples, "d2l", "d2l")
    all_metrics["d2l"] = {**m3, "sec_per_example": round(t3, 3)}

    # --- Сводная таблица ---
    print(f"\n{'='*60}")
    print("СВОДНАЯ ТАБЛИЦА: Qwen2.5-0.5B-IT на SQuAD (200 примеров)")
    print(f"{'='*60}")
    print(f"{'Вариант':<28} {'EM':>7} {'F1':>7} {'s/ex':>7}")
    print("-" * 52)
    for name, label in [("base", "(1) Base (no context)"),
                        ("context", "(2) +Context"),
                        ("d2l", "(3) D2L LoRA")]:
        m = all_metrics[name]
        print(f"  {label:<26} {m['EM']:>7.1f} {m['F1']:>7.1f} {m['sec_per_example']:>7.2f}")
    print("=" * 52)

    # Сохраняем
    output = {
        "model": config.model_name,
        "n_eval": N_EVAL,
        "checkpoint": checkpoint,
        **all_metrics,
        "examples_base": r1[:5],
        "examples_context": r2[:5],
        "examples_d2l": r3[:5],
    }
    out_path = RESULTS_DIR / "d2l_eval_squad.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nРезультаты сохранены: {out_path}")


if __name__ == "__main__":
    main()
