"""
DataLoader для обучения D2L на SQuAD.

Каждый пример:
  - context: параграф (вход encoder'а)
  - question + answer: для teacher (с контекстом) и student (без контекста)
"""

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

from .config import D2LConfig


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
        self.config = config

        ds = load_dataset("rajpurkar/squad", split=split)
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        self.data = ds

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        ex = self.data[idx]
        context  = ex["context"]
        question = ex["question"]
        answer   = ex["answers"]["text"][0] if ex["answers"]["text"] else ""

        # 1. Context tokens (для encoder'а)
        ctx_enc = self.tokenizer(
            context,
            truncation=True,
            max_length=self.config.max_chunk_len,
            return_tensors="pt",
        )

        # 2. Teacher prompt: context + question → answer
        teacher_text = (
            f"<|im_start|>system\nAnswer briefly based on the context.<|im_end|>\n"
            f"<|im_start|>user\nContext: {context}\n\nQuestion: {question}<|im_end|>\n"
            f"<|im_start|>assistant\n{answer}<|im_end|>"
        )
        teacher_enc = self.tokenizer(
            teacher_text,
            truncation=True,
            max_length=self.config.max_chunk_len,
            return_tensors="pt",
        )

        # 3. Student prompt: question only → answer (LoRA заменяет контекст)
        student_text = (
            f"<|im_start|>system\nAnswer briefly.<|im_end|>\n"
            f"<|im_start|>user\nQuestion: {question}<|im_end|>\n"
            f"<|im_start|>assistant\n{answer}<|im_end|>"
        )
        student_enc = self.tokenizer(
            student_text,
            truncation=True,
            max_length=self.config.max_chunk_len,
            return_tensors="pt",
        )

        # Labels: маскируем всё кроме ответа (-100)
        # Находим позицию "assistant\n" и маскируем до неё
        student_labels = student_enc["input_ids"].clone()
        assistant_marker = self.tokenizer.encode("assistant\n", add_special_tokens=False)
        # Простой подход: маскируем первые N токенов до ответа
        # Ищем последний "assistant" токен
        ids = student_labels[0].tolist()
        answer_start = len(ids)  # default: всё замаскировано
        for i in range(len(ids) - len(assistant_marker), -1, -1):
            if ids[i:i+len(assistant_marker)] == assistant_marker:
                answer_start = i + len(assistant_marker)
                break
        student_labels[0, :answer_start] = -100

        teacher_labels = teacher_enc["input_ids"].clone()
        t_ids = teacher_labels[0].tolist()
        t_answer_start = len(t_ids)
        for i in range(len(t_ids) - len(assistant_marker), -1, -1):
            if t_ids[i:i+len(assistant_marker)] == assistant_marker:
                t_answer_start = i + len(assistant_marker)
                break
        teacher_labels[0, :t_answer_start] = -100

        return {
            "ctx_input_ids":      ctx_enc["input_ids"].squeeze(0),
            "ctx_attention_mask":  ctx_enc["attention_mask"].squeeze(0),
            "teacher_input_ids":   teacher_enc["input_ids"].squeeze(0),
            "teacher_attention_mask": teacher_enc["attention_mask"].squeeze(0),
            "teacher_labels":      teacher_labels.squeeze(0),
            "student_input_ids":   student_enc["input_ids"].squeeze(0),
            "student_attention_mask": student_enc["attention_mask"].squeeze(0),
            "student_labels":      student_labels.squeeze(0),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate с padding (batch_size=1 обычно, но на всякий случай)."""
    result = {}
    for key in batch[0]:
        tensors = [b[key] for b in batch]
        max_len = max(t.shape[0] for t in tensors)
        padded = []
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
        num_workers=0,
    )


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from .config import auto_config

    cfg = auto_config()
    tok = AutoTokenizer.from_pretrained(cfg.model_name)

    dl = get_dataloader(tok, cfg, max_samples=5)
    batch = next(iter(dl))
    for k, v in batch.items():
        print(f"{k:30s} {v.shape}")
