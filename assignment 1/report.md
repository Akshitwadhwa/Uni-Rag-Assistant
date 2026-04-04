# Assignment-1: Retrieval-Augmented Generation for Course Handout Question Answering

## 1. Introduction

This project implements a Retrieval-Augmented Generation (RAG) system for domain-specific question answering in the academic domain. The selected use case is a **course handout assistant** where students upload subject handouts and ask factual questions such as credits, evaluation criteria, faculty details, and course topics. Instead of relying on the parametric memory of a general LLM, the system retrieves relevant content from uploaded handouts and generates answers grounded in that context.

The objective of the project is to reduce hallucination and improve answer reliability by combining document retrieval with LLM-based response generation.

## 2. Document Ingestion and Preprocessing

The knowledge base for this project consists of course handouts stored in [data/course_handouts](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/data/course_handouts). The handouts used in the final experiments were:

- `Generative AI and LLMs_CourseHandout.pdf`
- `Cryptography_CSE3703.pdf`
- `Course Handout - ToC (Jan-June).pdf`
- `IoT Networks, Architectures and Applications-CSE 2023 Batch.pdf`
- `2025_course_handout_AMCH.pdf`

The ingestion pipeline performs the following preprocessing steps:

- PDF reading using `pypdf`
- whitespace normalization
- cleanup of broken PDF spacing patterns
- metadata extraction from noisy inline headers
- extraction of course title, course code, credits, faculty, semester, and evaluation summary

This stage is implemented mainly in [ingestion.py](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment/ingestion.py).

The system was improved specifically to handle noisy PDF patterns such as:

- multiple metadata fields on one line
- broken words such as `A ssessment`
- irregular spacing between tokens

These improvements were important because several uploaded handouts were not cleanly structured after PDF extraction.

## 3. Text Chunking Strategy

The project uses **section-aware chunking** as the main chunking strategy. Since handouts often contain sections such as credits, evaluation pattern, faculty details, and topics, preserving section boundaries improves retrieval accuracy.

Three chunking ideas were considered:

| Chunking Strategy | Configuration | Observation |
|---|---|---|
| Fixed chunking | chunk size 500, overlap 80 | simple baseline |
| Sentence-window chunking | sentence-based windows | useful for narrative content, weaker for structured tables |
| Section-aware chunking | section-based + fixed fallback | best for course handouts |

In the implemented system, the main runs used:

- `chunk_strategy = section_fixed`
- `chunk_size = 500`
- `chunk_overlap = 80`

This strategy was chosen because:

- evaluation and credits often appear as compact sections
- section-aware chunking preserves these blocks more reliably
- topic-based questions also benefit from preserving the course content section

This logic is implemented in [chunking.py](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment/chunking.py).

## 4. Embedding Generation

Two Hugging Face embedding models were used for comparison:

| Embedding Model | Source | Dimension | Role |
|---|---|---:|---|
| `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face | 384 | lightweight baseline |
| `BAAI/bge-base-en-v1.5` | Hugging Face | 768 | stronger semantic retrieval |

The system supports different embedding models through [embeddings.py](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment/embeddings.py).

Observations:

- `MiniLM` was faster and lighter
- `BGE` produced stronger semantic retrieval for evaluation and topics-based questions
- the difference in dimension also showed that higher-dimensional embeddings improved some retrieval cases

Thus, the embedding comparison requirement of the assignment was satisfied through:

- different embedding models
- different embedding dimensions
- retrieval comparison across the same handouts and same questions

## 5. Vector Database Storage and Retrieval

Two vector databases were implemented:

| Vector Database | Purpose | Observation |
|---|---|---|
| `FAISS` | local dense similarity search | efficient and simple |
| `Chroma` | persistent vector database with metadata support | convenient for experiment management |

The vector database layer is implemented in [vectorstores.py](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment/vectorstores.py).

### Retrieval Workflow

1. each chunk is embedded
2. embeddings are stored in FAISS or Chroma
3. the user query is embedded
4. top relevant chunks are retrieved
5. additional ranking heuristics prioritize:
   - credits-related sections
   - evaluation-related sections
   - faculty-related sections
   - syllabus/topic sections

This ranking logic is implemented in [pipeline.py](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment/pipeline.py).

## 6. Prompt Engineering with Retrieved Context

Prompt engineering was used to ensure grounded and readable answers. The system prompt instructs the LLM to:

- answer only from retrieved course-handout context
- avoid unsupported claims
- preserve exact values such as marks, percentages, and credits
- write answers in plain English prose
- mention source files naturally

The retrieved chunks are formatted along with:

- subject name
- course code
- section title
- source filename

This helps the LLM understand the context provenance and answer more reliably. The prompt logic is implemented in [prompting.py](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment/prompting.py).

## 7. LLM-Based Answer Generation

The final answer generation was performed using hosted Hugging Face inference. The primary LLM used in the final retrieval experiments was:

- `meta-llama/Llama-3.1-8B-Instruct`

The system also supports multi-model comparison through `compare-models`, and a second model can be tested depending on provider availability. This logic is implemented in [generation.py](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment/generation.py).

For the current project, Llama 3.1 was used consistently in the retrieval comparison matrix so that embedding and vector database comparisons remained fair.

## 8. Comparative Evaluation of System Configurations

The main retrieval comparison matrix used in this project was:

- `FAISS + MiniLM + Llama`
- `FAISS + BGE + Llama`
- `Chroma + MiniLM + Llama`
- `Chroma + BGE + Llama`

The common practical question set used for final comparison was:

- What is the evaluation criteria for IoT Networks, Architectures and Applications?
- What topics are covered in IoT Networks, Architectures and Applications?
- What is the evaluation criteria for Applied Machine Learning in Health Care?
- How many credits are there in Applied Machine Learning in Health Care?

Additional supporting queries were also executed for:

- Generative AI and LLMs
- Theory of Computation
- Cryptography
- faculty/coordinator information

### Final Comparative Observations

| Configuration | Overall Observation |
|---|---|
| FAISS + MiniLM + Llama | good baseline, fast retrieval |
| FAISS + BGE + Llama | stronger semantic retrieval than MiniLM |
| Chroma + MiniLM + Llama | stable and reusable for experiments |
| Chroma + BGE + Llama | strongest and most consistent in the final shared question set |

Based on the final experiments, the best overall configuration was:

**Chroma + BGE + Llama**

This setup produced strong semantic retrieval, good consistency on the shared IoT and healthcare question set, and practical persistence for repeated experiments.

## 9. Example Output Demonstrating the System

The generated outputs were saved as JSON logs and can be used for screenshots in the submission. Result folders include:

- [FAISS_Outputs](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/FAISS_Outputs)
- [Chromo_Output](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/Chromo_Output)
- [Compare](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/Compare)

Recommended screenshots for the report:

- one indexing command execution
- one FAISS-based answer
- one Chroma-based answer
- one comparison table from the notebook
- one `compare-models` example if needed

Example system outputs include:

- credits answers
- evaluation criteria answers
- topic coverage answers
- faculty/coordinator answers

## 10. Experimental Analysis

A notebook was created to compare all major results:

- [results_comparison.ipynb](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/results_comparison.ipynb)

The notebook:

- loads all `ask` logs
- normalizes common question types
- builds side-by-side matrices for the major configurations
- loads `compare-models` logs
- exports summary CSVs for report usage

This notebook can be used to generate tables and screenshots for the final report.

## 11. Limitations

The current system has the following limitations:

- some PDFs still contain noisy extracted text
- tables and broken formatting reduce retrieval quality
- model availability depends on the Hugging Face provider support attached to the token
- no formal automatic evaluation metric such as exact match or F1 was used

## 12. Conclusion

This assignment demonstrates a successful implementation of a domain-specific RAG pipeline for course-handout question answering. The system was able to ingest real academic handouts, preprocess them, extract metadata, generate embeddings, index them using FAISS and Chroma, retrieve relevant content, and produce grounded answers using an LLM.

The experiments show that retrieval quality depends strongly on:

- chunking strategy
- embedding model
- vector database
- prompt design

Among the tested configurations, **Chroma + BGE + Llama** gave the strongest overall performance in the current experiment set.

## Appendix-1: Complete Source Code with Modular Design

The complete source code for the modular implementation is available in:

- [src/rag_assignment](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment)

Important modules:

- [ingestion.py](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment/ingestion.py)
- [chunking.py](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment/chunking.py)
- [embeddings.py](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment/embeddings.py)
- [vectorstores.py](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment/vectorstores.py)
- [prompting.py](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment/prompting.py)
- [generation.py](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment/generation.py)
- [pipeline.py](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment/pipeline.py)
- [cli.py](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment/cli.py)
