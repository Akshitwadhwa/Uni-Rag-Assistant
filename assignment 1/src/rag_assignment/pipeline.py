from __future__ import annotations

from dataclasses import dataclass
import re

from rag_assignment.chunking import Chunk, build_chunks
from rag_assignment.embeddings import TfidfEmbedder, create_embedder
from rag_assignment.generation import create_generator
from rag_assignment.ingestion import load_documents
from rag_assignment.prompting import build_prompt
from rag_assignment.vectorstores import SearchResult, create_vector_store


@dataclass
class RAGConfig:
    data_dir: str = "data/course_handouts"
    chunk_strategy: str = "section_fixed"
    chunk_size: int = 500
    chunk_overlap: int = 80
    sentence_window_size: int = 5
    sentence_stride: int = 3
    embedding_backend: str = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store: str = "faiss"
    generator_backend: str = "ollama"
    generator_model: str = "llama3.1"
    top_k: int = 6
    reranker_model: str | None = None


class RAGPipeline:
    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self.embedder = create_embedder(
            backend=config.embedding_backend,
            model_name=config.embedding_model,
        )
        self.vector_store = create_vector_store(config.vector_store)

    def ingest_and_index(self) -> list[Chunk]:
        documents = load_documents(self.config.data_dir)
        chunks = build_chunks(
            documents=documents,
            strategy=self.config.chunk_strategy,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            sentence_window_size=self.config.sentence_window_size,
            sentence_stride=self.config.sentence_stride,
        )

        if isinstance(self.embedder, TfidfEmbedder):
            self.embedder.fit([chunk.text for chunk in chunks])
        embeddings = self.embedder.embed_texts([chunk.text for chunk in chunks])
        if isinstance(self.embedder, TfidfEmbedder):
            self.embedder.save()
        self.vector_store.add(
            chunks=chunks,
            embeddings=embeddings,
            index_metadata={
                "embedding_backend": self.config.embedding_backend,
                "embedding_model": self.config.embedding_model,
                "vector_store": self.config.vector_store,
                "chunk_strategy": self.config.chunk_strategy,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
            },
        )
        return chunks

    def retrieve(self, question: str, top_k: int | None = None) -> list[SearchResult]:
        query_vector = self.embedder.embed_texts([question])[0]
        requested_k = top_k or self.config.top_k
        results = self.vector_store.search(query_embedding=query_vector, top_k=max(requested_k * 3, requested_k))
        results = self.rank_for_course_query(question, results)
        if self.config.reranker_model:
            results = self.rerank(question, results)
        return results[:requested_k]

    def rank_for_course_query(self, question: str, results: list[SearchResult]) -> list[SearchResult]:
        normalized_question = re.sub(r"\s+", " ", question.lower()).strip()
        ranked: list[SearchResult] = []

        for result in results:
            boosted_score = result.score
            subject_name = str(result.chunk.metadata.get("subject_name", "")).lower()
            course_code = str(result.chunk.metadata.get("course_code", "")).lower()
            section_type = str(result.chunk.metadata.get("section_type", "")).lower()

            if subject_name and subject_name in normalized_question:
                boosted_score += 0.35
            if course_code and course_code in normalized_question:
                boosted_score += 0.35

            if any(keyword in normalized_question for keyword in ["credit", "credits", "ltp", "contact hour"]):
                if section_type == "credits":
                    boosted_score += 0.3
            if any(keyword in normalized_question for keyword in ["evaluation", "assessment", "grading", "marks", "criteria"]):
                if section_type == "evaluation":
                    boosted_score += 0.3
            if any(keyword in normalized_question for keyword in ["syllabus", "unit", "module", "topic", "topics"]):
                if section_type == "syllabus":
                    boosted_score += 0.25
            if any(keyword in normalized_question for keyword in ["faculty", "instructor", "teacher", "coordinator"]):
                if section_type == "faculty":
                    boosted_score += 0.25

            ranked.append(SearchResult(chunk=result.chunk, score=boosted_score))

        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked

    def rerank(self, question: str, results: list[SearchResult]) -> list[SearchResult]:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("Install sentence-transformers to use reranking.") from exc

        reranker = CrossEncoder(self.config.reranker_model)
        pairs = [[question, result.chunk.text] for result in results]
        scores = reranker.predict(pairs)
        reranked = [
            SearchResult(chunk=result.chunk, score=float(score))
            for result, score in zip(results, scores, strict=False)
        ]
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked

    def answer(self, question: str) -> tuple[str, list[SearchResult]]:
        results = self.retrieve(question)
        prompt = build_prompt(question, results)
        generator = create_generator(
            backend=self.config.generator_backend,
            model_name=self.config.generator_model,
        )
        return generator.generate(prompt), results
