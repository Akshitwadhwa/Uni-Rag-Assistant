# Assignment 1: RAG for Domain-Specific Question Answering

This project implements a configurable Retrieval-Augmented Generation (RAG) system for a **Personal Course Handout Assistant**. A student uploads course handouts or syllabi for the subjects they are studying, and the system answers questions grounded in those documents.

The project satisfies the assignment rubric with:

- document ingestion and preprocessing
- multiple chunking strategies and chunk sizes
- structured metadata extraction from handouts
- multiple embedding backends
- two vector databases: `FAISS` and `Chroma`
- semantic retrieval with metadata-aware ranking
- multiple LLM backends for answer generation

## Domain Choice

The selected domain is **course handouts and subject syllabi**. This is a strong fit because the questions are factual and document-grounded:

- How many credits does Machine Learning carry?
- What is the evaluation criteria for Operating Systems?
- Does Computer Networks have a lab component?
- What topics are covered in Unit 4 of DBMS?

The included sample handouts cover:

- Database Management Systems
- Operating Systems
- Machine Learning
- Computer Networks

You can replace the contents of `data/course_handouts/` with your own PDFs or text handouts and reuse the same pipeline.

## Project Structure

```text
.
├── data/
│   ├── course_handouts/
│   │   ├── computer_networks_handout.md
│   │   ├── dbms_handout.md
│   │   ├── machine_learning_handout.md
│   │   └── operating_systems_handout.md
│   └── university_kb/
├── report_template.md
├── requirements.txt
└── src/
    └── rag_assignment/
        ├── __init__.py
        ├── chunking.py
        ├── cli.py
        ├── embeddings.py
        ├── generation.py
        ├── ingestion.py
        ├── pipeline.py
        ├── prompting.py
        └── vectorstores.py
```

## Setup

Use Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If you want to use API-based embeddings or chat models, create a `.env` from `.env.example`.

## Core Experiments

Recommended comparisons for the report:

1. Chunking strategies:
   - `section_fixed`
   - `fixed`
   - `sentence_window`
2. Chunk sizes: `300`, `500`, `800`
3. Chunk overlap: `50`, `100`
4. Embeddings:
   - `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
   - `BAAI/bge-base-en-v1.5` (768-dim)
   - `text-embedding-3-small` via OpenAI-compatible API (1536-dim)
5. Vector stores:
   - `faiss`
   - `chroma`
6. Generators:
   - `huggingface` hosted chat models
   - `ollama` with `llama3.1`
   - `openai`-compatible chat model if available

Why `section_fixed` matters here:

- handouts usually contain sections like `Evaluation Scheme`, `Credits`, and `Course Content`
- section-aware chunking keeps these blocks intact
- this improves answers for credits, marks distribution, and unit-wise syllabus queries

## Metadata Extraction

The pipeline extracts these fields from each handout:

- `course_code`
- `subject_name`
- `credits`
- `ltp`
- `faculty`
- `semester`
- `section_title`
- `section_type`

This makes direct factual queries more reliable than flat chunk retrieval alone.

## Example Commands

Build an index with a Hugging Face embedding model and FAISS:

```bash
PYTHONPATH=src python -m rag_assignment.cli index \
  --data-dir data/course_handouts \
  --vector-store faiss \
  --chunk-strategy section_fixed \
  --embedding-backend sentence-transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --chunk-size 500 \
  --chunk-overlap 80
```

Ask a question using Ollama:

```bash
PYTHONPATH=src python -m rag_assignment.cli ask \
  --question "What is the evaluation criteria for Operating Systems?" \
  --vector-store faiss \
  --chunk-strategy section_fixed \
  --embedding-backend sentence-transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --generator-backend huggingface \
  --generator-model meta-llama/Llama-3.1-8B-Instruct \
  --top-k 4
```

Compare two hosted models on the same query:

```bash
PYTHONPATH=src python -m rag_assignment.cli compare-models \
  --question "How many credits are there in Cryptography?" \
  --vector-store faiss \
  --chunk-strategy section_fixed \
  --embedding-backend sentence-transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --generator-backend huggingface \
  --generator-models meta-llama/Llama-3.1-8B-Instruct google/gemma-2-9b-it
```

Compare retrieval configurations:

```bash
PYTHONPATH=src python -m rag_assignment.cli compare \
  --question "How many credits are there in Machine Learning?" \
  --configs miniLM-faiss bge-chroma
```

## Example Questions

- How many credits are there in Database Management Systems?
- What is the evaluation criteria for Operating Systems?
- Does Computer Networks have a separate lab component?
- What topics are covered in Unit 3 of Machine Learning?
- Who is the instructor for CSC301?

## Suggested Report Flow

Use [report_template.md](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/report_template.md) as the base. The recommended structure is:

1. Problem statement
2. Why RAG is needed instead of plain prompting
3. Dataset and domain
4. Metadata extraction from handouts
5. System architecture
6. Chunking experiments
7. Embedding model comparison
8. Vector database comparison
9. LLM answer quality comparison
10. Limitations and future work

## Notes

- `FAISS` is typically faster for local dense retrieval experiments.
- `Chroma` is easier to inspect and persist and is a good fit for metadata-heavy workflows.
- If you do not want local inference, use `SentenceTransformers + FAISS/Chroma + Hugging Face hosted inference`.
- If Ollama is not installed, use the retrieval-only experiment results plus the code framework and discuss generator setup in the report.
