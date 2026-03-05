# Text-to-LoRA и Doc-to-LoRA: запуск и анализ

Курсовая работа по машинному обучению, ВМК МГУ, 2026.


---

## Оригинальные репозитории

| Метод | Репозиторий | Статья |
|-------|-------------|--------|
| Text-to-LoRA (T2L) | [SakanaAI/text-to-lora](https://github.com/SakanaAI/text-to-lora) | [arxiv 2506.06105](https://arxiv.org/abs/2506.06105)|
| Doc-to-LoRA (D2L) | [SakanaAI/doc-to-lora](https://github.com/SakanaAI/doc-to-lora) | [arxiv 2602.15902](https://arxiv.org/abs/2602.15902) |


## Результаты запуска

### Text-to-LoRA

`generate_lora.py` — выполнен успешно. Сгенерированы 3 LoRA-адаптера
(математика, логика, программирование).

`run_eval.py` — **не запускается на T4**: Gemma-2 attention logits soft capping
несовместим с XFormers backend в vLLM 0.5.4. Требуется A100.

Метрики (из README оригинального репозитория, gemma-2-2b-it):

| Модель | ArcC | ArcE | BoolQ | GSM8K | PIQA | AVG |
|--------|------|------|-------|-------|------|-----|
| base | 73.63 | 89.86 | 80.98 | 55.27 | 70.84 | 60.76 |
| + ICL | 72.10 | 88.80 | 82.29 | 55.27 | 67.68 | 63.45 |
| T2L + ICL | **74.12** | **90.03** | 82.27 | **56.61** | **73.61** | **66.09** |

### Doc-to-LoRA

`model.internalize(doc)` и `model.generate()` — работают корректно на T4.

`run_eval.py` — не запускается по той же причине что и T2L.

Метрики (из [pub.sakana.ai](https://pub.sakana.ai/doc-to-lora/), относительные):

| Метрика | Значение |
|---------|----------|
| SQuAD vs full-context upper bound | 83.5% |
| Long-context QA vs oracle CD | 85% (oracle = 90%, но 40 сек + >7 GB) |
| NIAH точность до ~40K токенов | near-perfect |
| Память vs KV-cache 128K doc | <50 MB vs >12 GB |

Абсолютные цифры воспроизводятся через `scripts/main_exp/eval/*.sh` (требуется A100).

---

## Краткое сравнение методов

|  | T2L | D2L |
|--|-----|-----|
| Вход | Описание задачи (промпт) | Произвольный документ |
| Энкодер | gte-large-en-v1.5 (внешняя модель) | Активации базовой LLM |
| Агрегатор | Linear projection | Perceiver Resampler (9 блоков) |
| Target modules | q_proj, v_proj (attention) | down_proj (MLP) |
| Что кодирует адаптер | Поведение (как отвечать) | Знания (что знать) |
| Параметры гиперсети | ~130M | ~315M |
| API | `generate_lora(task)` → файл | `model.internalize(doc)` |
