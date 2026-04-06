"""
Эксперимент 6: Text-to-LoRA (T2L) — генерация адаптеров из описания задачи
===========================================================================
Демонстрируем гиперсеть T2L (SakanaAI):
  1. Генерируем 4 LoRA-адаптера по текстовому описанию задачи:
       - Russian QA    (ответы на вопросы, русский)
       - English QA    (ответы на вопросы, английский)
       - NER           (именованные сущности)
       - Summarization (суммаризация диалогов)
  2. Загружаем базовую модель (gemma-2-2b-it)
  3. Для каждой задачи сравниваем:
       базовая модель  vs  базовая модель + T2L-адаптер

Требования:
  - Чекпоинт T2L: text-to-lora/trained_t2l/gemma_2b_t2l/
    (скачать: cd text-to-lora && uv run huggingface-cli download
              SakanaAI/text-to-lora --local-dir . --include "trained_t2l/gemma_2b_t2l/*")
  - ~10 GB RAM (модель gemma-2-2b-it)

Запуск (из директории text-to-lora/):
    uv run python ../t2l-d2l-study/experiments/06_text_to_lora.py
"""

import json
import os
import random
import string
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

# Путь к репозиторию text-to-lora (относительно этого файла ../../text-to-lora)
T2L_REPO = Path(__file__).parent.parent.parent / "text-to-lora"
T2L_DIR  = T2L_REPO / "trained_t2l" / "gemma_2b_t2l"

# model_loading.py ищет chat_templates/ относительно cwd — переходим в T2L_REPO
os.chdir(str(T2L_REPO))

# Добавляем src из text-to-lora в путь поиска модулей
_t2l_src = str(T2L_REPO / "src")
if _t2l_src not in sys.path:
    sys.path.insert(0, _t2l_src)

from peft import PeftModel, get_peft_config, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from hyper_llm_modulator.hyper_modulator import load_hypermod_checkpoint, save_lora
from hyper_llm_modulator.utils import get_layers, embed_texts

BASE_MODEL  = "google/gemma-2-2b-it"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Описания задач — в стиле обучающего датасета T2L
# ---------------------------------------------------------------------------

TASKS = {
    "russian_qa": (
        "Russian question answering",
        "This task requires you to answer questions in Russian based on a provided context paragraph. "
        "You must read the context carefully and extract the exact answer span from the text. "
        "Answer briefly using only information from the given context."
    ),
    "english_qa": (
        "English question answering",
        "This task requires you to answer questions in English based on a provided context paragraph. "
        "You must read the context carefully and extract a short, precise answer. "
        "Answer concisely using only information from the given context."
    ),
    "ner": (
        "Named entity recognition",
        "This task requires you to identify and classify named entities in English text. "
        "You must detect persons (PER), organizations (ORG), and locations (LOC). "
        "For each entity, output its text and its label in a structured format."
    ),
    "summarization": (
        "Dialogue summarization",
        "This task requires you to write a concise one-sentence summary of a dialogue. "
        "You must capture the main topic and outcome of the conversation. "
        "Be brief, factual, and do not include unnecessary details."
    ),
}

# Тестовые вопросы для каждой задачи
QUESTIONS = {
    "russian_qa": [
        "Контекст: Москва — столица России и крупнейший город страны. Население города превышает 12 миллионов человек. Москва является политическим, экономическим и культурным центром России.\n\nВопрос: Какое население Москвы?",
        "Контекст: Байкал — крупнейшее пресноводное озеро в мире по объёму воды. Оно расположено в Сибири и содержит около 20% мировых запасов поверхностных пресных вод.\n\nВопрос: Где находится озеро Байкал?",
    ],
    "english_qa": [
        "Context: The Amazon River is the largest river in the world by discharge volume of water. It is located in South America and flows through Brazil.\n\nQuestion: Where does the Amazon River flow?",
        "Context: Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.\n\nQuestion: Who created Python?",
    ],
    "ner": [
        "Identify named entities in: 'Apple CEO Tim Cook announced new products in San Francisco last Tuesday.'",
        "Identify named entities in: 'Barack Obama was born in Hawaii and served as the 44th President of the United States.'",
    ],
    "summarization": [
        "Summarize this dialogue:\nAlice: Hey, did you finish the report?\nBob: Yes, I sent it to the manager this morning.\nAlice: Great, what was the main finding?\nBob: Sales increased by 15% this quarter.",
        "Summarize this dialogue:\nMike: I can't make it to the meeting tomorrow.\nSarah: Why not?\nMike: I have a doctor's appointment at 10am.\nSarah: OK, I'll send you the notes afterwards.",
    ],
}


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 300) -> str:
    chat = [{"role": "user", "content": prompt}]
    # apply_chat_template в новых transformers возвращает BatchEncoding или тензор
    result = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(result, torch.Tensor):
        input_ids = result.to(model.device)
    else:
        input_ids = result["input_ids"].to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Шаг 1: Генерация LoRA-адаптеров через T2L (прямо в этом процессе)
# ---------------------------------------------------------------------------

def add_full_stop(s: str) -> str:
    s = s.strip()
    if s and s[-1].isalpha():
        s += "."
    return s


def generate_single_adapter(hypermod_state, task_desc: str, save_dir: Path) -> Path:
    """Генерирует один LoRA-адаптер по описанию задачи."""
    (_args, hypermod, model, _tokenizer,
     emb_model, emb_tokenizer,
     task_desc_format_fn, pooling_fn,
     peft_config, device) = hypermod_state

    layer_indices = torch.tensor(
        range(len(get_layers(model))), dtype=torch.long, device=device
    )

    task_emb = embed_texts(
        [task_desc], emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, device
    )
    encoder_out = hypermod.task_encoder(task_emb)
    encoded_task_emb = encoder_out["encoded_task_emb"].detach()
    lora_sd = hypermod.gen_lora(layer_indices, encoded_task_emb)

    save_dir.mkdir(parents=True, exist_ok=True)
    save_lora(lora_sd, peft_config, str(save_dir))
    (save_dir / "task_desc.txt").write_text(task_desc)
    return save_dir


def generate_adapters() -> dict[str, Path]:
    """Загружает T2L один раз и генерирует адаптеры для всех задач."""
    if not T2L_DIR.exists():
        print(f"\n[ERROR] Чекпоинт T2L не найден: {T2L_DIR}")
        print("Скачайте его:")
        print("  cd t2l-d2l-study")
        print("  uv run huggingface-cli download SakanaAI/text-to-lora \\")
        print(f"    --local-dir {T2L_REPO} --include 'trained_t2l/gemma_2b_t2l/*'")
        sys.exit(1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    checkpoint_path = T2L_DIR / "hypermod.pt"
    peft_config = get_peft_config(
        PeftConfig.from_json_file(str(T2L_DIR / "adapter_config.json"))
    )

    print(f"\n[T2L] Загружаю гиперсеть из {T2L_DIR.name}...")
    (args, hypermod, model, tokenizer,
     emb_model, emb_tokenizer,
     task_desc_format_fn, pooling_fn) = load_hypermod_checkpoint(
        str(checkpoint_path), device
    )
    hypermod.eval()

    hypermod_state = (
        args, hypermod, model, tokenizer,
        emb_model, emb_tokenizer,
        task_desc_format_fn, pooling_fn,
        peft_config, device
    )

    adapter_paths = {}
    out_dir = T2L_DIR / "extras" / "user_generated"

    for task_key, (task_name, task_desc) in TASKS.items():
        print(f"\n{'='*60}")
        print(f"[T2L] Генерирую адаптер: {task_name}")
        print(f"      Описание: {task_desc[:80]}...")
        print(f"{'='*60}")

        uid = "".join(random.choices(string.ascii_letters + string.digits, k=8))
        ts  = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        save_path = out_dir / f"{ts}_{uid}_{task_key}"

        generate_single_adapter(hypermod_state, add_full_stop(task_desc), save_path)
        adapter_paths[task_key] = save_path
        print(f"  Адаптер сохранён: {save_path.name}")

    # Освобождаем память от гиперсети (все ссылки из hypermod_state)
    del hypermod_state, hypermod, model, emb_model
    import gc; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return adapter_paths


# ---------------------------------------------------------------------------
# Шаг 2: Сравнение base vs adapted
# ---------------------------------------------------------------------------

def run_comparison(adapter_paths: dict[str, Path]) -> dict:
    device = get_device()
    print(f"\nDevice: {device}")
    print(f"Base model: {BASE_MODEL}")

    print(f"\n[Загружаю токенизатор {BASE_MODEL}]...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    all_results = {}

    for task_key, questions in QUESTIONS.items():
        adapter_path = adapter_paths[task_key]
        task_name = TASKS[task_key][0]

        print(f"\n{'='*60}")
        print(f"ЗАДАЧА: {task_name.upper()}")
        print(f"Адаптер: {adapter_path.name}")
        print(f"{'='*60}")

        # --- Базовая модель ---
        print("\nЗагружаю базовую модель...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map={"": str(device)},
            attn_implementation="eager",
        )
        base_model.eval()

        base_answers = []
        for q in questions:
            ans = generate(base_model, tokenizer, q)
            base_answers.append(ans)

        del base_model
        import gc; gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

        # --- Модель с T2L-адаптером ---
        print("Загружаю модель с T2L-адаптером...")
        adapted_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map={"": str(device)},
            attn_implementation="eager",
        )
        adapted_model = PeftModel.from_pretrained(
            adapted_model, str(adapter_path), is_trainable=False
        )
        adapted_model.eval()

        adapted_answers = []
        for q in questions:
            ans = generate(adapted_model, tokenizer, q)
            adapted_answers.append(ans)

        del adapted_model
        import gc; gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

        # --- Вывод ---
        task_results = []
        for i, (q, base_ans, adapted_ans) in enumerate(zip(questions, base_answers, adapted_answers)):
            print(f"\n[Вопрос {i+1}] {q[:120]}...")
            print(f"\n  БЕЗ адаптера:\n    {base_ans[:300]}")
            print(f"\n  С T2L-адаптером ({task_name}):\n    {adapted_ans[:300]}")
            print("-" * 50)
            task_results.append({
                "question": q,
                "base": base_ans,
                "adapted": adapted_ans,
            })

        all_results[task_key] = {
            "task_name": task_name,
            "adapter": str(adapter_path),
            "results": task_results,
        }

    return all_results


# ---------------------------------------------------------------------------
# Главный скрипт
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Text-to-LoRA (T2L) — генерация адаптеров из описания задачи")
    print("Базовая модель: google/gemma-2-2b-it")
    print("=" * 60)

    # Шаг 1: генерируем адаптеры (или берём уже готовые)
    # Чтобы пропустить генерацию передай SKIP_GEN=1:
    #   SKIP_GEN=1 uv run python experiments/06_text_to_lora.py
    if os.environ.get("SKIP_GEN"):
        user_gen = T2L_DIR / "extras" / "user_generated"
        existing = sorted(user_gen.iterdir()) if user_gen.exists() else []
        # берём последние 4 адаптера по алфавиту имени (они заканчиваются на _task_key)
        adapter_paths = {}
        for p in existing:
            for task_key in TASKS:
                if p.name.endswith(f"_{task_key}"):
                    adapter_paths[task_key] = p
        if len(adapter_paths) < len(TASKS):
            print("[WARN] Не все адаптеры найдены, запускаю генерацию...")
            adapter_paths = generate_adapters()
        else:
            print("[1] Используем ранее сгенерированные адаптеры:")
            for k, v in adapter_paths.items():
                print(f"  {k}: {v.name}")
    else:
        print("\n[1] Генерация LoRA-адаптеров через T2L гиперсеть...")
        adapter_paths = generate_adapters()

    print("\n\nСгенерированные адаптеры:")
    for task_key, path in adapter_paths.items():
        print(f"  {task_key}: {path}")

    # Шаг 2: сравнение
    print("\n[2] Сравнение base vs T2L-adapted...")
    results = run_comparison(adapter_paths)

    # Сохраняем
    out_path = RESULTS_DIR / "t2l_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n\nРезультаты сохранены: {out_path}")

    # Итоговая таблица
    print("\n" + "=" * 60)
    print("ИТОГ: T2L-адаптеры (gemma-2-2b-it)")
    print("=" * 60)
    for task_key, data in results.items():
        print(f"\n  {data['task_name'].upper()}")
        for r in data["results"]:
            print(f"    Q:       {r['question'][:60]}...")
            print(f"    Base:    {r['base'][:80]}")
            print(f"    T2L:     {r['adapted'][:80]}")


if __name__ == "__main__":
    main()
