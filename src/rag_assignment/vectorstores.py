from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rag_assignment.chunking import Chunk


@dataclass(slots=True)
class SearchResult:
    chunk: Chunk
    score: float


class BaseVectorStore:
    def add(self, chunks: list[Chunk], embeddings: np.ndarray, index_metadata: dict | None = None) -> None:
        raise NotImplementedError

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> list[SearchResult]:
        raise NotImplementedError


class FAISSStore(BaseVectorStore):
    def __init__(self, persist_dir: str = "artifacts/faiss_store") -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._chunks: list[Chunk] = []
        self._index = None
        self._index_metadata: dict = {}

    def add(self, chunks: list[Chunk], embeddings: np.ndarray, index_metadata: dict | None = None) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("Install faiss-cpu to use FAISS storage.") from exc

        vectors = np.asarray(embeddings, dtype="float32")
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        self._chunks = list(chunks)
        self._index_metadata = {
            **(index_metadata or {}),
            "embedding_dimension": int(vectors.shape[1]),
            "chunk_count": len(chunks),
        }

        faiss.write_index(self._index, str(self.persist_dir / "index.faiss"))
        np.save(self.persist_dir / "vectors.npy", vectors)
        metadata = [asdict(chunk) for chunk in self._chunks]
        (self.persist_dir / "chunks.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (self.persist_dir / "index_metadata.json").write_text(
            json.dumps(self._index_metadata, indent=2),
            encoding="utf-8",
        )

    def load(self) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("Install faiss-cpu to use FAISS storage.") from exc

        index_path = self.persist_dir / "index.faiss"
        chunk_path = self.persist_dir / "chunks.json"
        metadata_path = self.persist_dir / "index_metadata.json"
        if not index_path.exists() or not chunk_path.exists():
            raise FileNotFoundError("FAISS index files not found. Run the index command first.")

        self._index = faiss.read_index(str(index_path))
        records = json.loads(chunk_path.read_text(encoding="utf-8"))
        self._chunks = [Chunk(**record) for record in records]
        if metadata_path.exists():
            self._index_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> list[SearchResult]:
        if self._index is None:
            self.load()

        query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        expected_dimension = int(self._index.d)
        actual_dimension = int(query.shape[1])
        if actual_dimension != expected_dimension:
            embedding_model = self._index_metadata.get("embedding_model", "unknown")
            raise ValueError(
                "Embedding dimension mismatch for FAISS index. "
                f"Index expects dimension {expected_dimension}, but query used {actual_dimension}. "
                f"The saved index was built with embedding model '{embedding_model}'. "
                "Re-run the index command with the same --embedding-model and --vector-store before asking questions."
            )
        scores, indices = self._index.search(query, top_k)
        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            results.append(SearchResult(chunk=self._chunks[idx], score=float(score)))
        return results


class ChromaStore(BaseVectorStore):
    def __init__(self, persist_dir: str = "artifacts/chroma_store", collection_name: str = "rag_assignment") -> None:
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    def _connect(self) -> None:
        if self._collection is not None:
            return
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError("Install chromadb to use Chroma storage.") from exc

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(name=self.collection_name)

    def add(self, chunks: list[Chunk], embeddings: np.ndarray, index_metadata: dict | None = None) -> None:
        self._connect()
        existing = self._collection.get()
        if existing and existing.get("ids"):
            self._collection.delete(ids=existing["ids"])
        metadata_path = self.persist_dir / "index_metadata.json"
        index_details = {
            **(index_metadata or {}),
            "embedding_dimension": int(np.asarray(embeddings, dtype="float32").shape[1]),
            "chunk_count": len(chunks),
        }
        metadata_path.write_text(json.dumps(index_details, indent=2), encoding="utf-8")
        self._collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=np.asarray(embeddings, dtype="float32").tolist(),
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    **chunk.metadata,
                    "doc_id": chunk.doc_id,
                    "source": chunk.source,
                    "chunk_id": chunk.chunk_id,
                }
                for chunk in chunks
            ],
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> list[SearchResult]:
        self._connect()
        response = self._collection.query(
            query_embeddings=[np.asarray(query_embedding, dtype="float32").tolist()],
            n_results=top_k,
        )
        results: list[SearchResult] = []
        ids = response["ids"][0]
        docs = response["documents"][0]
        metas = response["metadatas"][0]
        distances = response["distances"][0]
        for chunk_id, text, meta, distance in zip(ids, docs, metas, distances, strict=False):
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=meta["doc_id"],
                text=text,
                source=meta["source"],
                metadata={k: v for k, v in meta.items() if k not in {"doc_id", "source"}},
            )
            score = 1.0 / (1.0 + float(distance))
            results.append(SearchResult(chunk=chunk, score=score))
        return results


def create_vector_store(name: str) -> BaseVectorStore:
    if name == "faiss":
        return FAISSStore()
    if name == "chroma":
        return ChromaStore()
    raise ValueError(f"Unsupported vector store: {name}")
