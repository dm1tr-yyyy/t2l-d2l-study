"""
DataLoader для обучения D2L на нескольких QA-датасетах.

Промпты из статьи D2L (Listing 7 и 8):
  Teacher: SELF_RESPONSE_TEMPLATE — контекст + вопрос как user message
  Student: QA_PROMPT — только вопрос как user message
  Ответ: gold answer (teacher forcing)

Поддерживаемые датасеты (DATASET_REGISTRY):
  squad  : rajpurkar/squad   (SQuAD v1.1, span extraction)
  drop   : ucinlp/drop       (DROP, берётся первый span-ответ)
  ropes  : allenai/ropes     (background + situation → контекст)
"""

import random

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

from .config import D2LConfig


# ---------------------------------------------------------------------------
# Промпты из статьи D2L (Listing 7, 8)
# ---------------------------------------------------------------------------

# Teacher: контекст + вопрос (user message)
SELF_RESPONSE_TEMPLATE = (
    "You are an honest and helpful assistant.\n\n"
    "# Provided Information\n"
    "{context}\n\n---\n\n"
    "# System Instruction\n"
    "- The information provided is up-to-date information and/or the user instruction.\n"
    "- When the provided information is not relevant to the question, ***ignore*** it "
    "and answer the question based on your knowledge.\n"
    "- If the provided information is related to the question, incorporate it in your response.\n"
    "- If the provided information is an instruction, follow the instruction carefully.\n"
    "\n---\n\n"
    "# User Input\n"
    "{question}"
)

# Student: только вопрос (user message), eval prompt из Listing 8
QA_PROMPT = (
    "Answer the following question. Output only the answer "
    "and do not output any other words.\n\nQuestion: {question}"
)


def _apply_template(tokenizer, user_content: str, answer: str, max_length: int, enable_thinking: bool = False):
    """
    Применяет chat template, возвращает input_ids и labels.
    Labels маскируют всё кроме токенов assistant-ответа.
    """
    messages = [
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": answer},
    ]

    # Полный текст с ответом
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].squeeze(0)

    # Длина prompt без ответа — маскируем всё до начала ответа
    prompt_text = tokenizer.apply_chat_template(
        [messages[0]],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    prompt_len = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]

    labels = input_ids.clone()
    labels[:prompt_len] = -100

    # Обрезаем до max_length
    if input_ids.shape[0] > max_length:
        input_ids = input_ids[:max_length]
        labels    = labels[:max_length]

    return input_ids, labels


# ---------------------------------------------------------------------------
# Адаптеры: HF-датасет → общий формат {"context", "question", "answer"}
# Возвращают None для примеров без валидного ответа — отфильтровываются.
# ---------------------------------------------------------------------------

def _adapt_squad(ex: dict) -> dict | None:
    answers = ex["answers"]["text"]
    if not answers:
        return None
    return {"context": ex["context"], "question": ex["question"], "answer": answers[0]}


def _adapt_drop(ex: dict) -> dict | None:
    """Берёт первый span-ответ. Числовые/датные ответы отбрасываются."""
    spans = ex["answers_spans"]["spans"]
    if not spans:
        return None
    return {"context": ex["passage"], "question": ex["question"], "answer": spans[0]}


def _adapt_ropes(ex: dict) -> dict | None:
    answers = ex["answers"]["text"]
    if not answers:
        return None
    return {
        "context": f"{ex['background']}\n\n{ex['situation']}",
        "question": ex["question"],
        "answer": answers[0],
    }


DATASET_REGISTRY: dict[str, tuple[str, callable]] = {
    "squad": ("rajpurkar/squad", _adapt_squad),
    "drop":  ("ucinlp/drop",     _adapt_drop),
    "ropes": ("allenai/ropes",   _adapt_ropes),
}


def _load_qa_split(name: str, split: str) -> list[dict]:
    """Загружает датасет и прогоняет через адаптер."""
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}"
        )
    hf_name, adapter = DATASET_REGISTRY[name]
    ds = load_dataset(hf_name, split=split)
    out: list[dict] = []
    for ex in ds:
        item = adapter(ex)
        if item is not None:
            item["_source"] = name
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# QADataset: универсальный (context, question, answer) → токенизированный sample
# ---------------------------------------------------------------------------

class QADataset(Dataset):
    """Универсальный QA-датасет для KL-дистилляции D2L."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: D2LConfig,
        samples: list[dict],
    ):
        self.tokenizer = tokenizer
        self.config    = config

        # Фильтр по длине контекста (батчевая токенизация для скорости)
        ctxs = [s["context"] for s in samples]
        ctx_lengths = tokenizer(
            ctxs,
            add_special_tokens=False,
            truncation=False,
            return_length=True,
        )["length"]
        keep = [i for i, l in enumerate(ctx_lengths) if l <= config.max_chunk_len]
        if len(keep) < len(samples):
            print(f"  Отфильтровано {len(samples) - len(keep)} длинных примеров "
                  f"(>{config.max_chunk_len} токенов), осталось {len(keep)}")
        self.samples = [samples[i] for i in keep]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s        = self.samples[idx]
        context  = s["context"]
        question = s["question"]
        answer   = s["answer"]

        # 1. Context tokens для encoder (plain tokenization, без chat template)
        ctx_enc = self.tokenizer(
            context,
            truncation=True,
            max_length=self.config.max_chunk_len,
            return_tensors="pt",
        )

        # 2. Teacher: SELF_RESPONSE_TEMPLATE (context + question) → answer
        teacher_prompt = SELF_RESPONSE_TEMPLATE.format(context=context, question=question)
        teacher_ids, teacher_labels = _apply_template(
            self.tokenizer, teacher_prompt, answer,
            max_length=self.config.max_teacher_len,
        )

        # 3. Student: QA_PROMPT (question only) → answer
        student_prompt = QA_PROMPT.format(question=question)
        student_ids, student_labels = _apply_template(
            self.tokenizer, student_prompt, answer,
            max_length=self.config.max_chunk_len,
        )

        return {
            "ctx_input_ids":          ctx_enc["input_ids"].squeeze(0),
            "ctx_attention_mask":     ctx_enc["attention_mask"].squeeze(0),
            "teacher_input_ids":      teacher_ids,
            "teacher_attention_mask": (teacher_ids != self.tokenizer.pad_token_id).long(),
            "teacher_labels":         teacher_labels,
            "student_input_ids":      student_ids,
            "student_attention_mask": (student_ids != self.tokenizer.pad_token_id).long(),
            "student_labels":         student_labels,
        }


def _pad_tensors(batch: list[dict], keys: list[str]) -> dict:
    """Right-padding для teacher/student тензоров."""
    result = {}
    for key in keys:
        tensors = [b[key] for b in batch]
        max_len = max(t.shape[0] for t in tensors)
        padded  = []
        for t in tensors:
            pad_len = max_len - t.shape[0]
            if pad_len > 0:
                pad_val = -100 if "labels" in key else 0
                t = torch.nn.functional.pad(t, (0, pad_len), value=pad_val)
            padded.append(t)
        result[key] = torch.stack(padded)
    return result


def packing_collate_fn(batch: list[dict], max_packed_ctx_len: int) -> dict:
    """
    Sequence packing: все ctx документы батча конкатенируются в одну последовательность.
    Teacher/student паддятся раздельно как обычно.

    Возвращает:
        packed_ctx_ids: [1, packed_len]  — все ctx без padding
        doc_lengths: list[int]           — длины каждого документа (для split после encoder)
        teacher_*/student_*: [N, max_len] — как раньше
    """
    ctx_ids = [b["ctx_input_ids"] for b in batch]
    doc_lengths = [ids.shape[0] for ids in ctx_ids]
    packed_len  = sum(doc_lengths)

    # Если суммарная длина превышает лимит — обрезаем хвост
    if packed_len > max_packed_ctx_len:
        keep, total = [], 0
        for length in doc_lengths:
            if total + length > max_packed_ctx_len:
                break
            keep.append(length)
            total += length
        if not keep:               # хотя бы один документ
            keep = [min(doc_lengths[0], max_packed_ctx_len)]
        ctx_ids    = ctx_ids[:len(keep)]
        doc_lengths = keep
        batch       = batch[:len(keep)]

    packed_ctx = torch.cat(ctx_ids).unsqueeze(0)  # [1, packed_len]

    result = _pad_tensors(batch, [
        "teacher_input_ids", "teacher_attention_mask", "teacher_labels",
        "student_input_ids", "student_attention_mask", "student_labels",
    ])
    result["packed_ctx_ids"] = packed_ctx
    result["doc_lengths"]    = doc_lengths
    return result


def get_dataloader(
    tokenizer: PreTrainedTokenizer,
    config: D2LConfig,
    split: str = "train",
    max_samples: int | None = None,
    datasets: list[str] | None = None,
    shuffle_seed: int = 42,
) -> DataLoader:
    """
    Args:
        datasets: имена датасетов из DATASET_REGISTRY (default: ["squad"]).
                  Все загружаются и конкатенируются в один поток.
        max_samples: ограничение общего числа примеров после конкатенации
                     (применяется после детерминированного shuffle, для тестов).
        shuffle_seed: seed для детерминированного pre-shuffle при max_samples.
    """
    if datasets is None:
        datasets = ["squad"]

    print(f"Загружаю датасеты {datasets} (split={split})")
    all_samples: list[dict] = []
    for name in datasets:
        samples = _load_qa_split(name, split)
        print(f"  {name}: {len(samples)} примеров")
        all_samples.extend(samples)
    print(f"Всего до фильтрации по длине: {len(all_samples)}")

    if max_samples is not None and len(all_samples) > max_samples:
        rng = random.Random(shuffle_seed)
        rng.shuffle(all_samples)
        all_samples = all_samples[:max_samples]
        print(f"  → обрезано до {max_samples}")

    ds = QADataset(tokenizer, config, all_samples)
    collate = lambda batch: packing_collate_fn(batch, config.max_packed_ctx_len)
    return DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
    )


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from .config import auto_config

    cfg = auto_config()
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    dl = get_dataloader(tok, cfg, max_samples=3, datasets=["squad", "drop", "ropes"])
    batch = next(iter(dl))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k:35s} {v.shape}")
        else:
            print(f"{k:35s} {v}")

    ex = dl.dataset[0]
    print("\nTeacher prompt (decoded):")
    print(tok.decode(ex["teacher_input_ids"][ex["teacher_labels"] != -100]))
    print("\nStudent prompt (decoded):")
    print(tok.decode(ex["student_input_ids"][ex["student_labels"] != -100]))
