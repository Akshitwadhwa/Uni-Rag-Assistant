from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    model_name: str

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        ...

    @property
    def dimension(self) -> int:
        ...


@dataclass
class SentenceTransformerEmbedder:
    model_name: str

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install sentence-transformers to use this embedder.") from exc

        self._model = SentenceTransformer(self.model_name)
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype="float32")

    @property
    def dimension(self) -> int:
        return self._dimension


@dataclass
class OpenAICompatibleEmbedder:
    model_name: str
    api_key: str | None = None
    base_url: str | None = None

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install openai to use this embedder.") from exc

        self._client = OpenAI(
            api_key=self.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=self.base_url or os.getenv("OPENAI_BASE_URL"),
        )
        self._dimension: int | None = None

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        response = self._client.embeddings.create(model=self.model_name, input=texts)
        vectors = np.asarray([item.embedding for item in response.data], dtype="float32")
        if self._dimension is None and len(vectors):
            self._dimension = int(vectors.shape[1])
        return vectors

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise ValueError("Embedding dimension is unknown until the first embedding call.")
        return self._dimension


@dataclass
class TfidfEmbedder:
    model_name: str = "tfidf-baseline"
    max_features: int = 2048
    persist_path: str = "artifacts/tfidf_vectorizer.pkl"

    def __post_init__(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise ImportError("Install scikit-learn to use this embedder.") from exc

        self._vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words="english")
        self._fitted = False
        self._dimension = self.max_features

    def fit(self, texts: list[str]) -> None:
        matrix = self._vectorizer.fit_transform(texts)
        self._dimension = int(matrix.shape[1])
        self._fitted = True

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        with open(self.persist_path, "wb") as handle:
            pickle.dump(self._vectorizer, handle)

    def load(self) -> None:
        with open(self.persist_path, "rb") as handle:
            self._vectorizer = pickle.load(handle)
        if hasattr(self._vectorizer, "vocabulary_"):
            self._dimension = len(self._vectorizer.vocabulary_)
            self._fitted = True

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            self.load()
        matrix = self._vectorizer.transform(texts)
        dense = matrix.toarray().astype("float32")
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense / norms

    @property
    def dimension(self) -> int:
        return self._dimension


def create_embedder(
    backend: str,
    model_name: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Embedder:
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name=model_name)
    if backend == "openai":
        return OpenAICompatibleEmbedder(model_name=model_name, api_key=api_key, base_url=base_url)
    if backend == "tfidf":
        return TfidfEmbedder(model_name=model_name)
    raise ValueError(f"Unsupported embedding backend: {backend}")
