from __future__ import annotations

import argparse
import json

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
    print(json.dumps(
        {
            "status": "ok",
            "documents_dir": config.data_dir,
            "vector_store": config.vector_store,
            "embedding_backend": config.embedding_backend,
            "embedding_model": config.embedding_model,
            "chunk_count": len(chunks),
            "chunk_strategy": config.chunk_strategy,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
        },
        indent=2,
    ))


def run_ask(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    pipeline = RAGPipeline(config)
    answer, results = pipeline.answer(args.question)
    print("\nAnswer:\n")
    print(answer)
    print("\nRetrieved chunks:\n")
    for result in results:
        print(f"- score={result.score:.4f} | source={result.chunk.metadata.get('filename', result.chunk.source)}")
        print(f"  text={result.chunk.text[:220]}...")


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
    print(json.dumps(outputs, indent=2))


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
        answer = generator.generate(prompt)
        outputs.append(
            {
                "model": model_name,
                "answer": answer,
                "sources": sorted({result.chunk.metadata.get("filename", result.chunk.source) for result in results}),
            }
        )

    print(json.dumps(
        {
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
        },
        indent=2,
    ))


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
