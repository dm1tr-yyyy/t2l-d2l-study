"""
DataLoader для обучения D2L на SQuAD.

Промпты из статьи D2L (Listing 7 и 8):
  Teacher: SELF_RESPONSE_TEMPLATE — контекст + вопрос как user message
  Student: QA_PROMPT — только вопрос как user message
  Ответ: gold answer из SQuAD (teacher forcing)
"""

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


class SQuADDataset(Dataset):
    """SQuAD для KL-дистилляции D2L."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: D2LConfig,
        split: str = "train",
        max_samples: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.config    = config

        ds = load_dataset("rajpurkar/squad", split=split)
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))

        # Отфильтровываем примеры с контекстом длиннее max_chunk_len токенов
        ctx_lengths = tokenizer(
            ds["context"],
            add_special_tokens=False,
            truncation=False,
            return_length=True,
        )["length"]
        keep = [i for i, l in enumerate(ctx_lengths) if l <= config.max_chunk_len]
        if len(keep) < len(ds):
            print(f"  SQuAD {split}: отфильтровано {len(ds) - len(keep)} длинных примеров "
                  f"(>{config.max_chunk_len} токенов), осталось {len(keep)}")
            ds = ds.select(keep)
        self.data = ds

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        ex       = self.data[idx]
        context  = ex["context"]
        question = ex["question"]
        answer   = ex["answers"]["text"][0] if ex["answers"]["text"] else ""

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


def collate_fn(batch: list[dict]) -> dict:
    """Collate с right-padding."""
    result = {}
    for key in batch[0]:
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


def get_dataloader(
    tokenizer: PreTrainedTokenizer,
    config: D2LConfig,
    split: str = "train",
    max_samples: int | None = None,
) -> DataLoader:
    ds = SQuADDataset(tokenizer, config, split, max_samples)
    return DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        collate_fn=collate_fn,
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

    dl = get_dataloader(tok, cfg, max_samples=3)
    batch = next(iter(dl))
    for k, v in batch.items():
        print(f"{k:35s} {v.shape}")

    # Показать один пример
    ex = dl.dataset[0]
    print("\nTeacher prompt (decoded):")
    print(tok.decode(ex["teacher_input_ids"][ex["teacher_labels"] != -100]))
    print("\nStudent prompt (decoded):")
    print(tok.decode(ex["student_input_ids"][ex["student_labels"] != -100]))
