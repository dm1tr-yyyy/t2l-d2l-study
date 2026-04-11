"""
Интерфейс инференса D2L: internalize(document) + generate(question).

model.internalize("The Aurora station was built in 2019...")
answer = model.generate("When was the Aurora station built?")
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import D2LConfig, auto_config
from .doc_to_lora import DocToLoRA
from .lora_injection import inject_lora, remove_lora, is_lora_injected


class DocToLoRAInference:
    """Inference API для D2L гиперсети."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        config: D2LConfig | None = None,
    ):
        if config is None:
            config = auto_config()
        self.config = config
        self.device = torch.device(config.device)

        # Загружаем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Загружаем базовую модель
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        ).to(self.device)
        self.base_model.eval()

        # Строим гиперсеть, переиспользуя base_model как encoder
        self.d2l = DocToLoRA(config, base_model=self.base_model)
        self.d2l.to(self.device)
        self.d2l.eval()

        if checkpoint_path is not None:
            self.d2l.load_checkpoint(checkpoint_path)

    @torch.no_grad()
    def internalize(self, document: str) -> None:
        """
        'Запекает' документ в LoRA-веса и инжектит их в base_model.
        После этого generate() отвечает на вопросы без документа в промпте.
        """
        # Убираем предыдущий адаптер если есть
        if is_lora_injected(self.base_model, self.config):
            remove_lora(self.base_model, self.config)

        # Токенизируем документ
        inputs = self.tokenizer(
            document, return_tensors="pt",
            truncation=True, max_length=self.config.max_chunk_len,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Генерируем LoRA
        lora_dict = self.d2l(inputs["input_ids"], inputs["attention_mask"])

        # Detach для инференса (не нужен grad)
        lora_dict = {k: v.detach() for k, v in lora_dict.items()}

        # Инжектим в base_model
        inject_lora(self.base_model, lora_dict, self.config)

    @torch.no_grad()
    def generate(
        self,
        question: str,
        max_new_tokens: int = 256,
    ) -> str:
        """Генерирует ответ. LoRA уже внутри модели (после internalize)."""
        # Промпт с пустым <think> блоком — Qwen3 пропускает thinking
        _NO_THINK = "<think>\n</think>\n"
        prompt_text = (
            f"<|im_start|>system\nAnswer briefly.<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n{_NO_THINK}"
        )
        input_ids = self.tokenizer(
            prompt_text, return_tensors="pt",
        )["input_ids"].to(self.device)

        output = self.base_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        new_tokens = output[0][input_ids.shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # Fallback: если thinking всё равно просочился
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        return text

    def reset(self) -> None:
        """Убирает LoRA, восстанавливает голую модель."""
        remove_lora(self.base_model, self.config)

    def generate_base(self, question: str, max_new_tokens: int = 256) -> str:
        """Генерирует ответ голой модели (без LoRA)."""
        was_injected = is_lora_injected(self.base_model, self.config)
        if was_injected:
            remove_lora(self.base_model, self.config)
        answer = self.generate(question, max_new_tokens)
        # Note: LoRA не восстанавливается — вызывающий должен вызвать internalize снова
        return answer

    def generate_with_context(
        self,
        document: str,
        question: str,
        max_new_tokens: int = 256,
    ) -> str:
        """Генерирует ответ с документом в промпте (baseline +context)."""
        was_injected = is_lora_injected(self.base_model, self.config)
        if was_injected:
            remove_lora(self.base_model, self.config)

        prompt = f"Context: {document}\n\nQuestion: {question}\n\nAnswer briefly."
        answer = self.generate(prompt, max_new_tokens)
        return answer
