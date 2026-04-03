from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from rag_assignment.pipeline import RAGConfig, RAGPipeline


PRESET_CONFIGS = {
    "miniLM-faiss": RAGConfig(
        embedding_backend="sentence-transformers",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        vector_store="faiss",
        chunk_strategy="section_fixed",
        chunk_size=500,
        chunk_overlap=80,
    ),
    "bge-chroma": RAGConfig(
        embedding_backend="sentence-transformers",
        embedding_model="BAAI/bge-base-en-v1.5",
        vector_store="chroma",
        chunk_strategy="section_fixed",
        chunk_size=500,
        chunk_overlap=80,
    ),
}


OUTPUT_DIR = Path("output")


def slugify(value: str, max_length: int = 50) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    if not normalized:
        normalized = "run"
    return normalized[:max_length]


def write_output_log(command: str, payload: dict, label: str | None = None) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{slugify(label)}" if label else ""
    path = OUTPUT_DIR / f"{command}_{timestamp}{suffix}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def normalize_model_answer(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"(?im)^\s*(answer|justification|sources)\s*:\s*", "", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = cleaned.replace(" \n", "\n")
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    return cleaned.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG assignment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", default="data/course_handouts")
    common.add_argument("--chunk-strategy", default="section_fixed", choices=["fixed", "sentence_window", "section_fixed"])
    common.add_argument("--chunk-size", type=int, default=500)
    common.add_argument("--chunk-overlap", type=int, default=80)
    common.add_argument("--sentence-window-size", type=int, default=5)
    common.add_argument("--sentence-stride", type=int, default=3)
    common.add_argument(
        "--embedding-backend",
        default="sentence-transformers",
        choices=["sentence-transformers", "openai", "tfidf"],
    )
    common.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    common.add_argument("--vector-store", default="faiss", choices=["faiss", "chroma"])
    common.add_argument("--top-k", type=int, default=4)
    common.add_argument("--reranker-model", default=None)

    subparsers.add_parser("index", parents=[common], help="Ingest documents and build a vector index")

    ask = subparsers.add_parser("ask", parents=[common], help="Answer a question using RAG")
    ask.add_argument("--question", required=True)
    ask.add_argument("--generator-backend", default="huggingface", choices=["ollama", "openai", "huggingface", "transformers"])
    ask.add_argument("--generator-model", default="llama3.1")

    compare = subparsers.add_parser("compare", help="Compare preset configurations on one query")
    compare.add_argument("--question", required=True)
    compare.add_argument("--configs", nargs="+", required=True, choices=sorted(PRESET_CONFIGS))

    compare_models = subparsers.add_parser("compare-models", parents=[common], help="Generate answers from multiple models for one query")
    compare_models.add_argument("--question", required=True)
    compare_models.add_argument("--generator-backend", default="huggingface", choices=["ollama", "openai", "huggingface", "transformers"])
    compare_models.add_argument("--generator-models", nargs="+", required=True)

    return parser


def config_from_args(args: argparse.Namespace) -> RAGConfig:
    return RAGConfig(
        data_dir=args.data_dir,
        chunk_strategy=args.chunk_strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        sentence_window_size=args.sentence_window_size,
        sentence_stride=args.sentence_stride,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        vector_store=args.vector_store,
        generator_backend=getattr(args, "generator_backend", "ollama"),
        generator_model=getattr(args, "generator_model", "llama3.1"),
        top_k=args.top_k,
        reranker_model=args.reranker_model,
    )


def run_index(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    pipeline = RAGPipeline(config)
    chunks = pipeline.ingest_and_index()
    payload = {
        "status": "ok",
        "documents_dir": config.data_dir,
        "vector_store": config.vector_store,
        "embedding_backend": config.embedding_backend,
        "embedding_model": config.embedding_model,
        "chunk_count": len(chunks),
        "chunk_strategy": config.chunk_strategy,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
    }
    log_path = write_output_log("index", payload, config.vector_store)
    print(json.dumps(payload, indent=2))
    print(f"\nLog saved to: {log_path}")


def run_ask(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    pipeline = RAGPipeline(config)
    answer, results = pipeline.answer(args.question)
    normalized_answer = normalize_model_answer(answer)
    payload = {
        "question": args.question,
        "generator_backend": args.generator_backend,
        "generator_model": args.generator_model,
        "vector_store": config.vector_store,
        "embedding_model": config.embedding_model,
        "answer": normalized_answer,
        "retrieved_chunks": [
            {
                "score": round(result.score, 4),
                "source": result.chunk.metadata.get("filename", result.chunk.source),
                "subject_name": result.chunk.metadata.get("subject_name", "unknown"),
                "section_title": result.chunk.metadata.get("section_title", "unknown"),
                "text": result.chunk.text,
            }
            for result in results
        ],
    }
    log_path = write_output_log("ask", payload, args.question)
    print("\nAnswer:\n")
    print(normalized_answer)
    print("\nRetrieved chunks:\n")
    for result in results:
        print(f"- score={result.score:.4f} | source={result.chunk.metadata.get('filename', result.chunk.source)}")
        print(f"  text={result.chunk.text[:220]}...")
    print(f"\nLog saved to: {log_path}")


def run_compare(args: argparse.Namespace) -> None:
    outputs = []
    for preset_name in args.configs:
        config = PRESET_CONFIGS[preset_name]
        pipeline = RAGPipeline(config)
        pipeline.ingest_and_index()
        results = pipeline.retrieve(args.question)
        outputs.append(
            {
                "config": preset_name,
                "vector_store": config.vector_store,
                "embedding_model": config.embedding_model,
                "top_results": [
                    {
                        "score": round(result.score, 4),
                        "source": result.chunk.metadata.get("filename", result.chunk.source),
                        "text_preview": result.chunk.text[:180],
                    }
                    for result in results
                ],
            }
        )
    payload = {"question": args.question, "comparisons": outputs}
    log_path = write_output_log("compare", payload, args.question)
    print(json.dumps(outputs, indent=2))
    print(f"\nLog saved to: {log_path}")


def run_compare_models(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    pipeline = RAGPipeline(config)
    results = pipeline.retrieve(args.question)

    outputs = []
    from rag_assignment.generation import create_generator
    from rag_assignment.prompting import build_prompt

    prompt = build_prompt(args.question, results)
    for model_name in args.generator_models:
        generator = create_generator(
            backend=args.generator_backend,
            model_name=model_name,
        )
        answer = normalize_model_answer(generator.generate(prompt))
        outputs.append(
            {
                "model": model_name,
                "answer": answer,
                "sources": sorted({result.chunk.metadata.get("filename", result.chunk.source) for result in results}),
            }
        )

    payload = {
        "question": args.question,
        "generator_backend": args.generator_backend,
        "retrieved_context": [
            {
                "score": round(result.score, 4),
                "subject_name": result.chunk.metadata.get("subject_name", "unknown"),
                "section_title": result.chunk.metadata.get("section_title", "unknown"),
                "source": result.chunk.metadata.get("filename", result.chunk.source),
            }
            for result in results
        ],
        "model_outputs": outputs,
    }
    log_path = write_output_log("compare_models", payload, args.question)
    print(json.dumps(payload, indent=2))
    print(f"\nLog saved to: {log_path}")


def main() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv()

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "index":
        run_index(args)
    elif args.command == "ask":
        run_ask(args)
    elif args.command == "compare":
        run_compare(args)
    elif args.command == "compare-models":
        run_compare_models(args)


if __name__ == "__main__":
    main()
