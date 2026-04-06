"""
Эксперимент 7: Сравнение трёх вариантов gemma-2-2b-it
======================================================
Честное сравнение на одной модели (gemma-2-2b-it):
  (1) Base       — без контекста / zero-shot
  (2) +Context   — с контекстом / few-shot
  (3) T2L LoRA   — T2L-адаптер (сгенерирован по описанию задачи)

Задачи:
  - Russian QA  (SberSQuAD)   → EM, F1
  - English QA  (SQuAD)       → EM, F1
  - NER         (CoNLL-2003)  → F1, Precision, Recall
  - Summarization (SAMSum)    → ROUGE-1, ROUGE-2, ROUGE-L

Запуск:
    uv run python experiments/07_t2l_eval.py
"""

import gc
import json
import os
import re
import string
import time
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Пути и конфигурация
# ---------------------------------------------------------------------------

SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
T2L_REPO    = SCRIPT_DIR.parent.parent / "text-to-lora"
T2L_GEN_DIR = T2L_REPO / "trained_t2l" / "gemma_2b_t2l" / "extras" / "user_generated"

# model_loading.py ищет chat_templates/ относительно cwd
os.chdir(str(T2L_REPO))

BASE_MODEL = "google/gemma-2-2b-it"
N_EVAL     = 200
N_FEW_SHOT = 3   # примеров в few-shot промпте для NER/Summ


# ---------------------------------------------------------------------------
# Адаптеры
# ---------------------------------------------------------------------------

def find_adapter(task_key: str) -> Path:
    candidates = sorted(
        [p for p in T2L_GEN_DIR.iterdir() if p.name.endswith(f"_{task_key}")],
        key=lambda p: p.name,
    )
    if not candidates:
        raise FileNotFoundError(
            f"T2L-адаптер для '{task_key}' не найден в {T2L_GEN_DIR}.\n"
            "Запустите сначала: uv run python experiments/06_text_to_lora.py"
        )
    return candidates[-1]


# ---------------------------------------------------------------------------
# Device / модель
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_base_model(device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map={"": str(device)},
        attn_implementation="eager",
    )
    model.eval()
    return model


def load_adapted_model(adapter_path: Path, device: torch.device):
    model = load_base_model(device)
    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model.eval()
    return model


def free_model(model, device: torch.device):
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Инференс
# ---------------------------------------------------------------------------

def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200,
             few_shot_turns: list[dict] | None = None) -> tuple[str, float]:
    """few_shot_turns: список {"user": ..., "assistant": ...} — вставляются перед финальным вопросом."""
    chat = []
    if few_shot_turns:
        for turn in few_shot_turns:
            chat.append({"role": "user",      "content": turn["user"]})
            chat.append({"role": "assistant", "content": turn["assistant"]})
    chat.append({"role": "user", "content": prompt})
    result = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt",
    )
    input_ids = (result if isinstance(result, torch.Tensor) else result["input_ids"]).to(model.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0

    new_tokens = output[0][input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, elapsed


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    return ' '.join(s.split())

def compute_em(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))

def compute_f1_qa(pred, gold):
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    common = Counter(pt) & Counter(gt)
    ns = sum(common.values())
    if ns == 0: return 0.0
    p = ns / len(pt); r = ns / len(gt)
    return 2 * p * r / (p + r)

NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

def decode_ner(tokens, tag_ids):
    entities = {"PER": [], "ORG": [], "LOC": []}
    cur, cur_type = [], None
    for tok, tag in zip(tokens, tag_ids):
        label = tag if isinstance(tag, str) else (NER_LABELS[tag] if tag < len(NER_LABELS) else "O")
        if label.startswith("B-"):
            if cur: entities[cur_type].append(" ".join(cur))
            cur = [tok]; cur_type = label[2:]
        elif label.startswith("I-") and cur_type == label[2:]:
            cur.append(tok)
        else:
            if cur: entities[cur_type].append(" ".join(cur))
            cur, cur_type = [], None
    if cur: entities[cur_type].append(" ".join(cur))
    return entities

def entities_to_string(entities):
    parts = [f"{t}: {', '.join(e)}" for t, e in entities.items() if e]
    return " | ".join(parts) if parts else "none"

def compute_ner_f1(pred_str, gold_entities):
    def parse(s):
        found = set()
        for part in s.split("|"):
            if ":" in part:
                et, rest = part.split(":", 1)
                et = et.strip().upper()
                for ent in rest.split(","):
                    ent = ent.strip().lower()
                    if ent and ent != "none":
                        found.add(f"{et}:{ent}")
        return found
    gold_set = {f"{t}:{e.lower()}" for t, ents in gold_entities.items() for e in ents}
    pred_set = parse(pred_str.lower())
    tp = len(pred_set & gold_set); fp = len(pred_set - gold_set); fn = len(gold_set - pred_set)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
    return {"f1": f1, "precision": p, "recall": r}

_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def compute_rouge(pred, gold):
    s = _rouge.score(gold, pred)
    return {"rouge1": s["rouge1"].fmeasure, "rouge2": s["rouge2"].fmeasure, "rougeL": s["rougeL"].fmeasure}


N_EXAMPLES = 2   # сколько полных примеров выводить после каждого прогона

def print_examples(examples: list[dict], task: str, mode: str):
    task_labels = {
        "russian_qa":    "Russian QA",
        "english_qa":    "English QA",
        "ner":           "NER",
        "summarization": "Summarization",
    }
    mode_labels = {"base": "Base (no context)", "context": "+Context / few-shot", "t2l": "T2L LoRA"}
    print(f"\n  ┌─ Примеры: {task_labels.get(task, task)} [{mode_labels.get(mode, mode)}]")
    for j, ex in enumerate(examples[:N_EXAMPLES]):
        print(f"  │  [{j+1}]")
        # Поле зависит от задачи
        if "question" in ex:
            print(f"  │   Вопрос : {ex['question'][:120]}")
        if "sentence" in ex:
            print(f"  │   Предл. : {ex['sentence'][:120]}")
        if "dialogue" in ex:
            # диалог может быть длинным — обрезаем
            dlg = ex["dialogue"].replace("\n", " / ")
            print(f"  │   Диалог : {dlg[:150]}")
        print(f"  │   Gold   : {ex['gold'][:120]}")
        print(f"  │   Pred   : {ex['pred'][:120]}")
        if j < N_EXAMPLES - 1:
            print(f"  │")
    print(f"  └{'─'*60}")


# ===========================================================================
# ЗАДАЧА 1: Russian QA (SberSQuAD)
# ===========================================================================

def load_russian_qa():
    ds = load_dataset("ERmak1581/QA_sberquad")["train"]
    return list(ds.select(range(len(ds) - N_EVAL, len(ds))))


def eval_russian_qa(model, tokenizer, examples, mode: str) -> dict:
    """mode: 'base' (без контекста) | 'context' | 't2l' (с контекстом)"""
    results, t_total = [], 0.0
    for i, ex in enumerate(examples):
        if mode == "base":
            prompt = (
                "Answer the following question in Russian briefly (1-5 words), "
                "only the answer, nothing else.\n\n"
                f"Question: {ex['Q']}"
            )
        else:
            prompt = (
                "Answer the question in Russian based on the context. "
                "Answer briefly (1-5 words), only the answer, nothing else.\n\n"
                f"Context: {ex['C']}\n\nQuestion: {ex['Q']}"
            )
        pred, elapsed = generate(model, tokenizer, prompt, max_new_tokens=50)
        t_total += elapsed
        gold = ex["A"]
        results.append({"question": ex["Q"], "gold": gold, "pred": pred,
                        "em": compute_em(pred, gold), "f1": compute_f1_qa(pred, gold)})
        if (i + 1) % 50 == 0:
            em_n = sum(r["em"] for r in results) / len(results) * 100
            f1_n = sum(r["f1"] for r in results) / len(results) * 100
            print(f"  {i+1}/{N_EVAL} — EM={em_n:.1f}, F1={f1_n:.1f}")

    em = sum(r["em"] for r in results) / len(results) * 100
    f1 = sum(r["f1"] for r in results) / len(results) * 100
    spe = t_total / len(results)
    print(f"  [{mode}] EM={em:.2f}, F1={f1:.2f} | {spe:.2f}s/ex")
    print_examples(results, "russian_qa", mode)
    return {"EM": round(em, 2), "F1": round(f1, 2), "sec_per_example": round(spe, 3),
            "n": len(results), "examples": results[:N_EXAMPLES]}


# ===========================================================================
# ЗАДАЧА 2: English QA (SQuAD)
# ===========================================================================

def load_english_qa():
    return list(load_dataset("rajpurkar/squad")["validation"].select(range(N_EVAL)))


def eval_english_qa(model, tokenizer, examples, mode: str) -> dict:
    results, t_total = [], 0.0
    for i, ex in enumerate(examples):
        gold = ex["answers"]["text"][0] if ex["answers"]["text"] else ""
        if mode == "base":
            prompt = (
                "Answer the following question briefly (1-5 words), "
                "only the answer, nothing else.\n\n"
                f"Question: {ex['question']}"
            )
        else:
            prompt = (
                "Answer the question based on the context. "
                "Answer briefly (1-5 words), only the answer, nothing else.\n\n"
                f"Context: {ex['context']}\n\nQuestion: {ex['question']}"
            )
        pred, elapsed = generate(model, tokenizer, prompt, max_new_tokens=50)
        t_total += elapsed
        results.append({"question": ex["question"], "gold": gold, "pred": pred,
                        "em": compute_em(pred, gold), "f1": compute_f1_qa(pred, gold)})
        if (i + 1) % 50 == 0:
            em_n = sum(r["em"] for r in results) / len(results) * 100
            f1_n = sum(r["f1"] for r in results) / len(results) * 100
            print(f"  {i+1}/{N_EVAL} — EM={em_n:.1f}, F1={f1_n:.1f}")

    em = sum(r["em"] for r in results) / len(results) * 100
    f1 = sum(r["f1"] for r in results) / len(results) * 100
    spe = t_total / len(results)
    print(f"  [{mode}] EM={em:.2f}, F1={f1:.2f} | {spe:.2f}s/ex")
    print_examples(results, "english_qa", mode)
    return {"EM": round(em, 2), "F1": round(f1, 2), "sec_per_example": round(spe, 3),
            "n": len(results), "examples": results[:N_EXAMPLES]}


# ===========================================================================
# ЗАДАЧА 3: NER (CoNLL-2003)
# ===========================================================================

NER_SYSTEM = (
    "You are a Named Entity Recognition (NER) assistant. "
    "Find all named entities in the sentence and list them by type. "
    "Types: PER (person), ORG (organization), LOC (location). "
    "Format: 'PER: name1, name2 | ORG: org1 | LOC: loc1'. "
    "If no entities of a type, skip it. If no entities at all, write 'none'. "
    "Output ONLY the entity list in the exact format above, nothing else."
)


def load_ner():
    return list(load_dataset("Davlan/conll2003_noMISC")["validation"].select(range(N_EVAL)))


def eval_ner(model, tokenizer, examples, mode: str) -> dict:
    # few-shot примеры — оформляем как отдельные user/assistant туры
    few_shot_turns = None
    if mode in ("context", "t2l"):
        shots_ds = load_dataset("Davlan/conll2003_noMISC")["train"]
        shots = []
        for ex in shots_ds:
            if len(ex["tokens"]) >= 5 and any(t != "O" for t in ex["ner_tags"]):
                shots.append(ex)
            if len(shots) == N_FEW_SHOT:
                break
        few_shot_turns = []
        for ex in shots:
            sent = " ".join(ex["tokens"])
            gold = entities_to_string(decode_ner(ex["tokens"], ex["ner_tags"]))
            few_shot_turns.append({"user": f"{NER_SYSTEM}\n\nSentence: {sent}",
                                   "assistant": gold})

    results, t_total = [], 0.0
    for i, ex in enumerate(examples):
        sentence      = " ".join(ex["tokens"])
        gold_entities = decode_ner(ex["tokens"], ex["ner_tags"])
        prompt = f"{NER_SYSTEM}\n\nSentence: {sentence}"

        pred, elapsed = generate(model, tokenizer, prompt, max_new_tokens=100,
                                 few_shot_turns=few_shot_turns)
        t_total += elapsed
        metrics = compute_ner_f1(pred, gold_entities)
        results.append({"sentence": sentence, "gold": entities_to_string(gold_entities),
                        "pred": pred, **metrics})
        if (i + 1) % 50 == 0:
            f1_n = sum(r["f1"] for r in results) / len(results) * 100
            print(f"  {i+1}/{N_EVAL} — F1={f1_n:.1f}")

    f1   = sum(r["f1"]        for r in results) / len(results) * 100
    prec = sum(r["precision"] for r in results) / len(results) * 100
    rec  = sum(r["recall"]    for r in results) / len(results) * 100
    spe  = t_total / len(results)
    print(f"  [{mode}] F1={f1:.2f}, P={prec:.2f}, R={rec:.2f} | {spe:.2f}s/ex")
    print_examples(results, "ner", mode)
    return {"F1": round(f1, 2), "Precision": round(prec, 2), "Recall": round(rec, 2),
            "sec_per_example": round(spe, 3), "n": len(results), "examples": results[:N_EXAMPLES]}


# ===========================================================================
# ЗАДАЧА 4: Summarization (SAMSum)
# ===========================================================================

def load_summarization():
    return list(load_dataset("spencer/samsum_reformat")["test"].select(range(N_EVAL)))


def eval_summarization(model, tokenizer, examples, mode: str) -> dict:
    _sum_instr = ("Write a brief one-sentence summary of the following dialogue. "
                  "Be concise and factual.")
    few_shot_turns = None
    if mode in ("context", "t2l"):
        shots_ds = load_dataset("spencer/samsum_reformat")["train"].select(range(2))
        few_shot_turns = [
            {"user": f"{_sum_instr}\n\nDialogue:\n{ex['dialogue']}",
             "assistant": ex["summary"]}
            for ex in shots_ds
        ]

    results, t_total = [], 0.0
    for i, ex in enumerate(examples):
        prompt = f"{_sum_instr}\n\nDialogue:\n{ex['dialogue']}"
        pred, elapsed = generate(model, tokenizer, prompt, max_new_tokens=80,
                                 few_shot_turns=few_shot_turns)
        t_total += elapsed
        gold = ex["summary"]
        metrics = compute_rouge(pred, gold)
        results.append({"dialogue": ex["dialogue"], "gold": gold, "pred": pred, **metrics})
        if (i + 1) % 50 == 0:
            r1_n = sum(r["rouge1"] for r in results) / len(results) * 100
            print(f"  {i+1}/{N_EVAL} — ROUGE-1={r1_n:.1f}")

    r1 = sum(r["rouge1"] for r in results) / len(results) * 100
    r2 = sum(r["rouge2"] for r in results) / len(results) * 100
    rl = sum(r["rougeL"] for r in results) / len(results) * 100
    spe = t_total / len(results)
    print(f"  [{mode}] R1={r1:.2f}, R2={r2:.2f}, RL={rl:.2f} | {spe:.2f}s/ex")
    print_examples(results, "summarization", mode)
    return {"ROUGE-1": round(r1, 2), "ROUGE-2": round(r2, 2), "ROUGE-L": round(rl, 2),
            "sec_per_example": round(spe, 3), "n": len(results), "examples": results[:N_EXAMPLES]}


# ===========================================================================
# Вывод таблиц
# ===========================================================================

def print_gemma_table(results: dict):
    """Таблица gemma: base → +context → T2L"""
    print("\n" + "=" * 72)
    print("GEMMA-2-2B-IT: base → +context → T2L LoRA")
    print("=" * 72)

    # QA
    print("\n── QA (EM / F1) ─────────────────────────────────────────────────")
    print(f"{'Вариант':<28} {'RU EM':>7} {'RU F1':>7} {'EN EM':>7} {'EN F1':>7} {'s/ex':>6}")
    print("-" * 72)
    for mode, label in [("base", "(1) Base (no context)"),
                        ("context", "(2) +Context"),
                        ("t2l", "(3) T2L LoRA")]:
        rqa = results.get("russian_qa", {}).get(mode, {})
        eqa = results.get("english_qa", {}).get(mode, {})
        ru_em = f"{rqa['EM']:.1f}" if rqa else "—"
        ru_f1 = f"{rqa['F1']:.1f}" if rqa else "—"
        en_em = f"{eqa['EM']:.1f}" if eqa else "—"
        en_f1 = f"{eqa['F1']:.1f}" if eqa else "—"
        spe   = f"{rqa.get('sec_per_example', 0):.2f}" if rqa else "—"
        print(f"  {label:<26} {ru_em:>7} {ru_f1:>7} {en_em:>7} {en_f1:>7} {spe:>6}")

    # NER
    print("\n── NER (F1 / P / R) ─────────────────────────────────────────────")
    print(f"{'Вариант':<28} {'F1':>7} {'P':>7} {'R':>7} {'s/ex':>6}")
    print("-" * 56)
    for mode, label in [("base", "(1) Base (zero-shot)"),
                        ("context", "(2) Few-shot"),
                        ("t2l", "(3) T2L LoRA")]:
        m = results.get("ner", {}).get(mode, {})
        f1  = f"{m['F1']:.1f}"        if m else "—"
        p   = f"{m['Precision']:.1f}" if m else "—"
        r   = f"{m['Recall']:.1f}"    if m else "—"
        spe = f"{m.get('sec_per_example', 0):.2f}" if m else "—"
        print(f"  {label:<26} {f1:>7} {p:>7} {r:>7} {spe:>6}")

    # Summarization
    print("\n── Summarization (ROUGE) ────────────────────────────────────────")
    print(f"{'Вариант':<28} {'R-1':>7} {'R-2':>7} {'R-L':>7} {'s/ex':>6}")
    print("-" * 62)
    for mode, label in [("base", "(1) Base (zero-shot)"),
                        ("context", "(2) Few-shot"),
                        ("t2l", "(3) T2L LoRA")]:
        m = results.get("summarization", {}).get(mode, {})
        r1  = f"{m['ROUGE-1']:.1f}" if m else "—"
        r2  = f"{m['ROUGE-2']:.1f}" if m else "—"
        rl  = f"{m['ROUGE-L']:.1f}" if m else "—"
        spe = f"{m.get('sec_per_example', 0):.2f}" if m else "—"
        print(f"  {label:<26} {r1:>7} {r2:>7} {rl:>7} {spe:>6}")

    print("\n" + "=" * 72)


# ===========================================================================
# Главный скрипт
# ===========================================================================

TASK_LOADERS = {
    #"russian_qa":    (load_russian_qa,    eval_russian_qa),
    #"english_qa":    (load_english_qa,    eval_english_qa),
    #"ner":           (load_ner,           eval_ner),
    "summarization": (load_summarization, eval_summarization)
}


def main():
    device = get_device()
    print(f"Device: {device}  |  Base model: {BASE_MODEL}")

    # Находим T2L-адаптеры
    adapters = {}
    for key in TASK_LOADERS:
        adapters[key] = find_adapter(key)
    print("T2L-адаптеры найдены:")
    for k, p in adapters.items():
        print(f"  {k}: {p.name}")

    # Загружаем данные один раз
    print("\nЗагружаю датасеты...")
    datasets = {key: loader() for key, (loader, _) in TASK_LOADERS.items()}

    # Загружаем токенизатор один раз
    print(f"Загружаю токенизатор {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    all_results = {key: {} for key in TASK_LOADERS}

    # -----------------------------------------------------------------------
    # Шаг 1: base + context — один проход голой модели для всех задач
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("ШАГ 1: Голая модель (base + context) — все задачи")
    print(f"{'='*60}")
    print("Загружаю gemma-2-2b-it (без адаптера)...")
    base_model = load_base_model(device)

    for key, (_, eval_fn) in TASK_LOADERS.items():
        print(f"\n  Задача: {key}")
        for mode in ("context", "base"):
            print(f"    Режим: {mode}")
            all_results[key][mode] = eval_fn(base_model, tokenizer, datasets[key], mode)

    free_model(base_model, device)
    print("\n  [base model freed]")

    # -----------------------------------------------------------------------
    # Шаг 2: T2L-адаптер — по одной задаче за раз
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("ШАГ 2: T2L LoRA — по задаче")
    print(f"{'='*60}")

    for key, (_, eval_fn) in TASK_LOADERS.items():
        print(f"\n  Задача: {key}  |  Адаптер: {adapters[key].name}")
        print(f"  Загружаю gemma + T2L-адаптер...")
        model = load_adapted_model(adapters[key], device)
        all_results[key]["t2l"] = eval_fn(model, tokenizer, datasets[key], "t2l")
        free_model(model, device)
        print(f"  [adapter freed]")

    # Сохраняем
    out_path = RESULTS_DIR / "gemma_eval.json"
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\nРезультаты сохранены: {out_path}")

    # Итоговая таблица
    print_gemma_table(all_results)


if __name__ == "__main__":
    main()
