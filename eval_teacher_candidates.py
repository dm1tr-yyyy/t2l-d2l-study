"""
Оценка кандидатов в teacher-модели на 50 примерах SQuAD (validation).

Запуск:
    python eval_teacher_candidates.py
    python eval_teacher_candidates.py --n 100
"""

import argparse
import gc
import re
import string
import time
import traceback
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── Модели для сравнения ──────────────────────────────────────────────────
CANDIDATES = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]


# ─── Метрики ──────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    return ' '.join(s.split())


def em(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def f1(pred: str, gold: str) -> float:
    pt = normalize(pred).split()
    gt = normalize(gold).split()
    common = Counter(pt) & Counter(gt)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    p = ns / len(pt)
    r = ns / len(gt)
    return 2 * p * r / (p + r)


# ─── Генерация ─────────────────────────────────────────────────────────────

def generate_with_context(model, tokenizer, device, context, question, max_new_tokens=100):
    chat = [
        {"role": "system", "content": "Answer briefly based on the context."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
    ]
    result = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt",
    )
    if not isinstance(result, torch.Tensor):
        result = result["input_ids"]
    input_ids = result.to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ─── Eval одной модели ─────────────────────────────────────────────────────

def eval_model(model_name: str, examples: list[dict], device: torch.device) -> dict:
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    print("  Загружаю модель...", end=" ", flush=True)
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    print(f"{time.time() - t_load:.1f}s")

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Параметры: {n_params:.2f}B")

    em_scores, f1_scores = [], []
    t_total = 0.0

    for i, ex in enumerate(examples):
        context  = ex["context"]
        question = ex["question"]
        gold     = ex["answers"]["text"][0] if ex["answers"]["text"] else ""

        t0 = time.perf_counter()
        pred = generate_with_context(model, tokenizer, device, context, question)
        t_total += time.perf_counter() - t0

        em_scores.append(em(pred, gold))
        f1_scores.append(f1(pred, gold))

        if (i + 1) % 10 == 0:
            avg_em = sum(em_scores) / len(em_scores) * 100
            avg_f1 = sum(f1_scores) / len(f1_scores) * 100
            spe = t_total / len(em_scores)
            print(f"  [{i+1}/{len(examples)}] EM={avg_em:.1f}  F1={avg_f1:.1f}  ({spe:.2f}s/ex)")

    avg_em = sum(em_scores) / len(em_scores) * 100
    avg_f1 = sum(f1_scores) / len(f1_scores) * 100
    spe = t_total / len(em_scores)

    # Примеры
    print(f"\n  Примеры:")
    for ex, pred_text in zip(examples[:3], [
        generate_with_context(model, tokenizer, device, ex["context"], ex["question"])
        for ex in examples[:3]
    ]):
        gold = ex["answers"]["text"][0]
        print(f"    Q:    {ex['question'][:70]}")
        print(f"    Gold: {gold[:70]}")
        print(f"    Pred: {pred_text[:70]}")
        print()

    # Выгружаем модель
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "params_b": round(n_params, 2),
        "EM": round(avg_em, 2),
        "F1": round(avg_f1, 2),
        "s_per_ex": round(spe, 3),
    }


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Число примеров SQuAD")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Список моделей (по умолчанию CANDIDATES)")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    models = args.models or CANDIDATES

    print(f"\nЗагружаю SQuAD validation ({args.n} примеров)...")
    ds = load_dataset("rajpurkar/squad")["validation"]
    examples = list(ds.select(range(args.n)))

    results = []
    for model_name in models:
        try:
            r = eval_model(model_name, examples, device)
            results.append(r)
        except Exception as e:
            print(f"  ОШИБКА: {e}")
            traceback.print_exc()
            results.append({"model": model_name, "params_b": "?", "EM": "err", "F1": "err", "s_per_ex": "?"})

    # Итоговая таблица
    print(f"\n{'='*65}")
    print("ИТОГОВАЯ ТАБЛИЦА: F1 с контекстом на SQuAD validation")
    print(f"{'='*65}")
    print(f"  {'Модель':<35} {'Params':>6} {'EM':>6} {'F1':>6} {'s/ex':>6}")
    print(f"  {'-'*60}")
    for r in results:
        print(f"  {r['model']:<35} {str(r['params_b']):>6}B {str(r['EM']):>6} {str(r['F1']):>6} {str(r['s_per_ex']):>6}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
