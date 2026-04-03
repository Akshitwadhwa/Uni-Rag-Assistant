from __future__ import annotations

import os
from dataclasses import dataclass

import requests


class BaseGenerator:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


@dataclass
class OllamaGenerator(BaseGenerator):
    model_name: str
    base_url: str | None = None

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/generate"
        response = requests.post(
            url,
            json={"model": self.model_name, "prompt": prompt, "stream": False},
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["response"].strip()


@dataclass
class OpenAIChatGenerator(BaseGenerator):
    model_name: str
    api_key: str | None = None
    base_url: str | None = None

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install openai to use this generator.") from exc

        self._client = OpenAI(
            api_key=self.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=self.base_url or os.getenv("OPENAI_BASE_URL"),
        )

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()


@dataclass
class HuggingFaceChatGenerator(BaseGenerator):
    model_name: str
    api_key: str | None = None
    base_url: str | None = None

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install openai to use this generator.") from exc

        token = self.api_key or os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("Set HF_TOKEN in your environment or .env file to use Hugging Face hosted inference.")

        self._client = OpenAI(
            api_key=token,
            base_url=self.base_url or os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1"),
        )

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()


@dataclass
class TransformersGenerator(BaseGenerator):
    model_name: str

    def __post_init__(self) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError("Install transformers to use this generator.") from exc

        self._pipe = pipeline("text-generation", model=self.model_name)

    def generate(self, prompt: str) -> str:
        outputs = self._pipe(prompt, max_new_tokens=256, do_sample=False)
        return outputs[0]["generated_text"][len(prompt) :].strip()


def create_generator(
    backend: str,
    model_name: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> BaseGenerator:
    if backend == "ollama":
        return OllamaGenerator(model_name=model_name, base_url=base_url)
    if backend == "openai":
        return OpenAIChatGenerator(model_name=model_name, api_key=api_key, base_url=base_url)
    if backend == "huggingface":
        return HuggingFaceChatGenerator(model_name=model_name, api_key=api_key, base_url=base_url)
    if backend == "transformers":
        return TransformersGenerator(model_name=model_name)
    raise ValueError(f"Unsupported generator backend: {backend}")
