"""
Демонстрация D2L: экзотический документ → QA.

Показываем три варианта:
  1. LLM без адаптера (галлюцинирует)
  2. LLM + документ в контексте (baseline)
  3. LLM + D2L-адаптер (без документа в контексте)

Запуск:
    uv run python -m d2l_framework.demo [checkpoint.pt]
"""

import sys
import time

from .inference import DocToLoRAInference
from .config import auto_config


# Экзотический документ, который LLM точно не знает
EXOTIC_DOCUMENT = """
The Valmiera Protocol is a 2024 international framework for regulating autonomous
underwater mining drones in the Baltic Sea. Signed on March 15, 2024 in Valmiera, Latvia,
by representatives of Estonia, Latvia, Lithuania, Finland, and Sweden, the protocol
establishes a 200-meter exclusion zone around all protected coral formations.

The protocol was proposed by Dr. Elina Ozolina, a marine biologist at the University of
Latvia, after her 2023 study revealed that autonomous mining drones had damaged 34% of
deep-sea coral formations in the Gulf of Riga. The Valmiera Protocol requires all mining
operators to install acoustic deterrent devices (ADDs) on their drones and to submit
quarterly environmental impact reports.

Enforcement is handled by the Baltic Marine Coordination Center (BMCC), headquartered in
Tallinn, Estonia. The first violation was recorded in June 2024 when the Swedish mining
company NordDeep AB was fined €2.3 million for operating drones without ADDs near the
Irbe Strait protected zone.
"""

QUESTIONS = [
    "What is the Valmiera Protocol?",
    "Who proposed the Valmiera Protocol?",
    "What percentage of coral was damaged?",
    "Where is the BMCC headquartered?",
    "How much was NordDeep AB fined?",
]


def main():
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else None

    config = auto_config()
    print("=" * 60)
    print("D2L Demo: экзотический документ → QA")
    print(f"Model: {config.model_name}")
    if checkpoint:
        print(f"Checkpoint: {checkpoint}")
    else:
        print("(без обученного чекпоинта — гиперсеть случайная)")
    print("=" * 60)

    model = DocToLoRAInference(
        checkpoint_path=checkpoint,
        config=config,
    )

    print(f"\nДокумент ({len(EXOTIC_DOCUMENT.split())} слов):")
    print(EXOTIC_DOCUMENT[:200] + "...")

    # Вариант 1: Base (без контекста, без LoRA)
    print("\n" + "=" * 60)
    print("ВАРИАНТ 1: Base (без документа)")
    print("=" * 60)
    for q in QUESTIONS:
        t0 = time.perf_counter()
        ans = model.generate(q, max_new_tokens=300)
        dt = time.perf_counter() - t0
        print(f"\n  Q: {q}")
        print(f"  A: {ans}")
        print(f"  ({dt:.2f}s)")

    # Вариант 2: +Context
    print("\n" + "=" * 60)
    print("ВАРИАНТ 2: +Context (документ в промпте)")
    print("=" * 60)
    for q in QUESTIONS:
        t0 = time.perf_counter()
        ans = model.generate_with_context(EXOTIC_DOCUMENT, q, max_new_tokens=300)
        dt = time.perf_counter() - t0
        print(f"\n  Q: {q}")
        print(f"  A: {ans}")
        print(f"  ({dt:.2f}s)")

    # Вариант 3: D2L
    print("\n" + "=" * 60)
    print("ВАРИАНТ 3: D2L (документ → LoRA)")
    print("=" * 60)

    t0 = time.perf_counter()
    model.internalize(EXOTIC_DOCUMENT)
    dt_intern = time.perf_counter() - t0
    print(f"  internalize: {dt_intern:.2f}s")

    for q in QUESTIONS:
        t0 = time.perf_counter()
        ans = model.generate(q, max_new_tokens=300)
        dt = time.perf_counter() - t0
        print(f"\n  Q: {q}")
        print(f"  A: {ans}")
        print(f"  ({dt:.2f}s)")

    model.reset()
    print("\n" + "=" * 60)
    print("Demo завершено.")


if __name__ == "__main__":
    main()
