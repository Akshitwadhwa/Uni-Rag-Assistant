# Retrieval-Augmented Generation for Domain-Specific Question Answering

## 1. Objective

The goal of this assignment is to build a Retrieval-Augmented Generation system for a domain-specific question answering task. The implemented system uses an external knowledge base, retrieves relevant chunks using vector similarity search, and injects the retrieved context into an LLM prompt before answer generation.

## 2. Selected Domain

This project uses a **Personal Course Handout Assistant** domain. The knowledge base contains course handouts or syllabi for the subjects a student is currently studying. Each handout includes details such as course code, subject title, credits, evaluation scheme, syllabus units, and faculty information.

Why this domain was selected:

- the content is structured and document-driven
- the queries are realistic and information-seeking
- factual grounding is important, so RAG offers a clear benefit over plain LLM prompting
- metadata extraction can be combined with semantic retrieval for higher reliability

## 3. System Architecture

The pipeline contains the following stages:

1. Document ingestion and preprocessing
2. Structured metadata extraction
3. Text chunking
4. Embedding generation
5. Vector database storage
6. Semantic retrieval and ranking
7. Prompt construction
8. LLM-based answer generation

## 4. Dataset and Knowledge Base

Describe the data sources used. For example:

- DBMS handout
- Operating Systems handout
- Machine Learning handout
- Computer Networks handout

Mention preprocessing:

- PDF or text extraction
- whitespace normalization while preserving section boundaries
- metadata extraction for course code, title, credits, semester, instructor
- metadata tagging with source filename, chunk id, and section type

## 5. Metadata Extraction

This project extracts key attributes from each handout before indexing.

Suggested fields:

| Field | Purpose |
|---|---|
| `course_code` | identify the subject precisely |
| `subject_name` | support natural-language subject queries |
| `credits` | answer direct factual queries |
| `ltp` | expose lecture-tutorial-practical breakup |
| `semester` | allow future filtering |
| `faculty` | answer faculty-related questions |
| `section_title` | preserve document structure |
| `section_type` | improve retrieval for evaluation, credits, syllabus, etc. |

Discuss why metadata-aware retrieval is better than only using raw chunks for this domain.

## 6. Chunking Strategy

Test multiple chunk sizes and overlaps, but also compare structure-aware chunking.

Suggested table:

| Strategy | Chunk Size | Overlap | Observation |
|---|---:|---:|---|
| Section-aware + fixed | 300 | 50 | Good for headings like credits and evaluation |
| Section-aware + fixed | 500 | 80 | Best balance for course handouts |
| Fixed only | 500 | 80 | Can split evaluation tables or key facts |
| Sentence window | N/A | N/A | Useful for narrative sections, weaker for structured fields |

Discussion points:

- smaller chunks improve precision but may miss surrounding context
- larger chunks improve completeness but may reduce retrieval sharpness
- section-aware chunking preserves factual handout blocks better than naive chunking

## 7. Embedding Models

Compare at least two embedding models.

Suggested comparison:

| Model | Source | Dimension | Notes |
|---|---|---:|---|
| all-MiniLM-L6-v2 | Hugging Face | 384 | Fast and lightweight |
| bge-base-en-v1.5 | Hugging Face | 768 | Better semantic retrieval, higher cost |
| text-embedding-3-small | OpenAI-compatible | 1536 | Strong retrieval quality, requires API access |

Discuss:

- retrieval relevance
- latency
- memory use
- effect of embedding dimension

## 8. Vector Database Comparison

Use at least two vector databases.

Suggested comparison:

| Vector DB | Advantages | Limitations |
|---|---|---|
| FAISS | Fast local dense search, efficient for experiments | Lower-level persistence workflow |
| Chroma | Easy persistence and metadata filtering | Slightly heavier runtime |

## 9. Prompt Engineering

The answer generation prompt should:

- instruct the model to answer only from retrieved context
- mention uncertainty when the context is insufficient
- cite source chunks where possible
- preserve exact values for credits, marks, and percentages

Example design:

> You are a course handout assistant. Use only the provided context. If the answer is not present, say that the information is not available in the uploaded handouts.

## 10. LLM Comparison

Compare different generators.

Suggested table:

| Model | Backend | Strengths | Weaknesses |
|---|---|---|---|
| LLaMA 3.1 | Ollama | Good reasoning and fluency | Local compute cost |
| Gemma 3 | Ollama | Efficient, concise | May be less detailed |
| OpenAI-compatible chat model | API | Strong answer quality | Requires internet/API |

## 11. Sample Queries

Include 5 to 10 evaluation questions such as:

- How many credits are there in Machine Learning?
- What is the evaluation criteria for Operating Systems?
- Does Computer Networks have a separate laboratory component?
- What topics are covered in Unit 4 of DBMS?
- Who is the instructor for CSC301?

## 12. Results and Analysis

Summarize which combination worked best.

Example:

- Best chunking setup: section-aware chunking with size 500 and overlap 80
- Best embedding model: `bge-base-en-v1.5`
- Best vector DB: FAISS for speed, Chroma for usability
- Best generator: LLaMA 3.1 with grounded prompt

## 13. Limitations

- quality depends on knowledge base quality
- metadata extraction may fail on poorly formatted PDFs
- local LLM performance depends on hardware
- no formal human evaluation in this baseline

## 14. Future Work

- add hybrid retrieval with BM25 + dense search
- add reranking with a cross-encoder
- support richer PDF table extraction
- add a web upload interface for student handouts

## 15. Conclusion

State that the RAG system reduced hallucination risk by grounding answers in uploaded subject handouts and that retrieval quality depends strongly on structured chunking, metadata extraction, embeddings, and vector database selection.
