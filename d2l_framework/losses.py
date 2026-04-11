"""
KL-дистилляция: sparse top-K KL divergence.

Из reference trainer.py:
  Teacher (с контекстом, frozen) → top-K logprobs
  Student (без контекста, с LoRA) → logits
  Loss = -sum_k p_k * log(q_k)

Sparse: teacher даёт только top-K токенов → эффективно по памяти.
"""

import torch
import torch.nn.functional as F


def compute_teacher_topk(
    logits: torch.Tensor,
    labels: torch.Tensor,
    top_k: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Предвычисляем top-K log-вероятности teacher'а.

    Args:
        logits: [batch, seq, vocab] — teacher logits
        labels: [batch, seq] — labels (-100 = ignore)
        top_k: количество токенов

    Returns:
        topk_logprobs: [N, K] — log-вероятности top-K токенов
        topk_indices:  [N, K] — индексы этих токенов
        где N — число позиций с labels != -100
    """
    # Shift: предсказание для позиции t делается из logits позиции t-1
    shift_logits = logits[:, :-1, :]  # [batch, seq-1, vocab]
    shift_labels = labels[:, 1:]       # [batch, seq-1]

    # Маска: только позиции с реальными labels
    mask = shift_labels != -100
    valid_logits = shift_logits[mask]  # [N, vocab]

    if valid_logits.numel() == 0:
        # Пустой ответ — возвращаем dummy
        device = logits.device
        return (torch.zeros(1, top_k, device=device),
                torch.zeros(1, top_k, dtype=torch.long, device=device))

    # Top-K (на CPU — MPS иногда падает на topk с большим vocab)
    logprobs = F.log_softmax(valid_logits.float().cpu(), dim=-1)
    topk_logprobs, topk_indices = logprobs.topk(top_k, dim=-1)

    return topk_logprobs.detach().to(logits.device), topk_indices.detach().to(logits.device)


def kl_distillation_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Sparse top-K KL divergence loss.

    Args:
        student_logits: [batch, seq, vocab]
        labels: [batch, seq] (-100 = ignore)
        teacher_topk_logprobs: [N, K] — от compute_teacher_topk
        teacher_topk_indices:  [N, K]

    Returns:
        loss: scalar
    """
    # Shift student logits так же
    shift_logits = student_logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    valid_logits = shift_logits[mask].float()  # [N, vocab]

    if valid_logits.numel() == 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

    # Выравниваем: teacher и student могут иметь разное число answer-токенов
    # (subword tokenization зависит от контекста перед ответом)
    N_student = valid_logits.shape[0]
    N_teacher = teacher_topk_logprobs.shape[0]
    N = min(N_student, N_teacher)
    if N == 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True)
    valid_logits = valid_logits[:N]
    teacher_topk_logprobs = teacher_topk_logprobs[:N]
    teacher_topk_indices  = teacher_topk_indices[:N]

    # Full log-partition (на CPU для стабильности на MPS)
    log_Z = torch.logsumexp(valid_logits.cpu(), dim=-1, keepdim=True).to(valid_logits.device)  # [N, 1]

    # Gather student logits at teacher's top-K indices
    student_at_topk = valid_logits.gather(1, teacher_topk_indices)  # [N, K]

    # Student log-probs at top-K positions
    student_logprobs = student_at_topk - log_Z  # [N, K]

    # Teacher probs (exp of log-probs)
    teacher_probs = teacher_topk_logprobs.exp()  # [N, K]

    # KL: -sum_k p_k * log(q_k)   (cross-entropy, positive loss)
    loss = -(teacher_probs * student_logprobs).sum(dim=-1)  # [N]

    return loss.mean()


def l1_regularization(lora_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """L1 регуляризация на сгенерированные LoRA-веса."""
    norm = lora_dict["A"].abs().mean() + lora_dict["B"].abs().mean()
    return norm
