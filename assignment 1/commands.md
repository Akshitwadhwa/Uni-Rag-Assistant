# Live Demo Commands

Use these commands from the Assignment 1 folder.

## 1. Setup

```bash
cd "/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1"
source ../.venv/bin/activate
```

## 2. FAISS Demo

### 2.1 Index with FAISS + BGE

```bash
PYTHONPATH=src python -m rag_assignment.cli index \
  --data-dir data/course_handouts \
  --vector-store faiss \
  --chunk-strategy section_fixed \
  --embedding-backend sentence-transformers \
  --embedding-model BAAI/bge-base-en-v1.5 \
  --chunk-size 500 \
  --chunk-overlap 80
```

### 2.2 Simple factual question

```bash
PYTHONPATH=src python -m rag_assignment.cli ask \
  --question "How many credits are there in Applied Machine Learning in Health Care?" \
  --vector-store faiss \
  --chunk-strategy section_fixed \
  --embedding-backend sentence-transformers \
  --embedding-model BAAI/bge-base-en-v1.5 \
  --generator-backend huggingface \
  --generator-model meta-llama/Llama-3.1-8B-Instruct \
  --top-k 5
```

### 2.3 Good section-specific question

```bash
PYTHONPATH=src python -m rag_assignment.cli ask \
  --question "What is the evaluation criteria for Generative AI and LLMs?" \
  --vector-store faiss \
  --chunk-strategy section_fixed \
  --embedding-backend sentence-transformers \
  --embedding-model BAAI/bge-base-en-v1.5 \
  --generator-backend huggingface \
  --generator-model meta-llama/Llama-3.1-8B-Instruct \
  --top-k 5
```

### 2.4 Good topics question

```bash
PYTHONPATH=src python -m rag_assignment.cli ask \
  --question "What topics are covered in IoT Networks, Architectures and Applications?" \
  --vector-store faiss \
  --chunk-strategy section_fixed \
  --embedding-backend sentence-transformers \
  --embedding-model BAAI/bge-base-en-v1.5 \
  --generator-backend huggingface \
  --generator-model meta-llama/Llama-3.1-8B-Instruct \
  --top-k 5
```

### 2.5 Complex two-model comparison

```bash
PYTHONPATH=src python -m rag_assignment.cli compare-models \
  --question "Compare the evaluation criteria of Generative AI and LLMs, IoT Networks, and Applied Machine Learning in Health Care. Mention the weightage of quizzes, assignments, projects, practical work, and end-sem evaluation for each course." \
  --vector-store faiss \
  --chunk-strategy section_fixed \
  --embedding-backend sentence-transformers \
  --embedding-model BAAI/bge-base-en-v1.5 \
  --generator-backend huggingface \
  --generator-models meta-llama/Llama-3.1-8B-Instruct openai/gpt-oss-120b \
  --top-k 8
```

## 3. Chroma Demo

### 3.1 Index with Chroma + MiniLM

```bash
PYTHONPATH=src python -m rag_assignment.cli index \
  --data-dir data/course_handouts \
  --vector-store chroma \
  --chunk-strategy section_fixed \
  --embedding-backend sentence-transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --chunk-size 500 \
  --chunk-overlap 80
```

### 3.2 Simple factual question

```bash
PYTHONPATH=src python -m rag_assignment.cli ask \
  --question "Who is the instructor or coordinator for Applied Machine Learning in Health Care?" \
  --vector-store chroma \
  --chunk-strategy section_fixed \
  --embedding-backend sentence-transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --generator-backend huggingface \
  --generator-model meta-llama/Llama-3.1-8B-Instruct \
  --top-k 5
```

### 3.3 Good comparison question

```bash
PYTHONPATH=src python -m rag_assignment.cli ask \
  --question "Compare Cryptography and Network Security in terms of course focus and major topics." \
  --vector-store chroma \
  --chunk-strategy section_fixed \
  --embedding-backend sentence-transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --generator-backend huggingface \
  --generator-model meta-llama/Llama-3.1-8B-Instruct \
  --top-k 8
```

### 3.4 Complex multi-course topic comparison

```bash
PYTHONPATH=src python -m rag_assignment.cli compare-models \
  --question "Compare the topics covered in Generative AI and LLMs, Making Causal Inferences, and Applied Machine Learning in Health Care. Highlight the overlap and the distinct focus of each course." \
  --vector-store chroma \
  --chunk-strategy section_fixed \
  --embedding-backend sentence-transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --generator-backend huggingface \
  --generator-models meta-llama/Llama-3.1-8B-Instruct openai/gpt-oss-120b \
  --top-k 8
```

### 3.5 Complex reasoning question

```bash
PYTHONPATH=src python -m rag_assignment.cli compare-models \
  --question "Across all uploaded handouts, which courses appear more theory-heavy and which appear more application-oriented? Use topics, credits, practical work, project components, and evaluation pattern to justify the comparison." \
  --vector-store chroma \
  --chunk-strategy section_fixed \
  --embedding-backend sentence-transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --generator-backend huggingface \
  --generator-models meta-llama/Llama-3.1-8B-Instruct openai/gpt-oss-120b \
  --top-k 10
```

## 4. Best Live Demo Flow

If you want a short but impressive live demo, use this order:

1. Run one `index` command
2. Run one simple factual `ask` command
3. Run one evaluation-based `ask` command
4. Run one `compare-models` command with a complex multi-course comparison

Recommended minimal sequence:

```bash
PYTHONPATH=src python -m rag_assignment.cli index --data-dir data/course_handouts --vector-store faiss --chunk-strategy section_fixed --embedding-backend sentence-transformers --embedding-model BAAI/bge-base-en-v1.5 --chunk-size 500 --chunk-overlap 80
```

```bash
PYTHONPATH=src python -m rag_assignment.cli ask --question "How many credits are there in Applied Machine Learning in Health Care?" --vector-store faiss --chunk-strategy section_fixed --embedding-backend sentence-transformers --embedding-model BAAI/bge-base-en-v1.5 --generator-backend huggingface --generator-model meta-llama/Llama-3.1-8B-Instruct --top-k 5
```

```bash
PYTHONPATH=src python -m rag_assignment.cli ask --question "What is the evaluation criteria for Generative AI and LLMs?" --vector-store faiss --chunk-strategy section_fixed --embedding-backend sentence-transformers --embedding-model BAAI/bge-base-en-v1.5 --generator-backend huggingface --generator-model meta-llama/Llama-3.1-8B-Instruct --top-k 5
```

```bash
PYTHONPATH=src python -m rag_assignment.cli compare-models --question "Compare the evaluation criteria of Generative AI and LLMs, IoT Networks, and Applied Machine Learning in Health Care. Mention the weightage of quizzes, assignments, projects, practical work, and end-sem evaluation for each course." --vector-store faiss --chunk-strategy section_fixed --embedding-backend sentence-transformers --embedding-model BAAI/bge-base-en-v1.5 --generator-backend huggingface --generator-models meta-llama/Llama-3.1-8B-Instruct openai/gpt-oss-120b --top-k 8
```

## 5. Note

If `openai/gpt-oss-120b` is not supported by your Hugging Face provider access, replace it with another supported second model from the Hugging Face Playground and keep the rest of the command unchanged.
