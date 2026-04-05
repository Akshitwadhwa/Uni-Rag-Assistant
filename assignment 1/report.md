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

The final answer generation stage was implemented through hosted Hugging Face inference. In the current project, two LLMs were used in the generator comparison experiments:

- `meta-llama/Llama-3.1-8B-Instruct`
- `openai/gpt-oss-120b`

The generator layer is implemented in [generation.py](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment/generation.py), while multi-model comparisons are executed through the `compare-models` command in [cli.py](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment/cli.py).

### Role of Each LLM in the Experiments

- **Llama 3.1** was used as the fixed generator in the main retrieval comparison matrix so that FAISS vs Chroma and MiniLM vs BGE could be compared fairly.
- **GPT-OSS 120B** was used as the second generator in `compare-models` experiments to compare answer style, completeness, and grounding on the same retrieved context.

### Observations from LLM Comparison

Based on the saved comparison logs in [Compare](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/Compare) and [Complex_Commands](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/Complex_Commands):

- `meta-llama/Llama-3.1-8B-Instruct` generally produced more direct and concise responses.
- `openai/gpt-oss-120b` often generated more explanatory or polished comparative answers.
- For simple factual queries, both models worked reasonably well when retrieval quality was good.
- For multi-course comparison questions, the quality of the retrieved context had a strong impact on both models.

Thus, the LLM comparison requirement of the assignment was satisfied by comparing two different hosted open models on the same retrieved context.

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

## 10. Types of Questions Asked to the Model

To evaluate the RAG system properly, the prompts were not limited to only one kind of question. Instead, the model was tested using a spectrum of question types, ranging from simple factual extraction to complex multi-course comparisons. This helped assess not only answer correctness but also retrieval quality, section relevance, and multi-document reasoning ability.

### 10.1 Basic Factual Questions

These questions were designed to test whether the system could retrieve and answer a single explicit fact from one handout.

Examples:

- How many credits are there in Theory of Computation?
- How many credits are there in Applied Machine Learning in Health Care?
- Who is the instructor or coordinator for Applied Machine Learning in Health Care?

These questions are useful for validating:

- metadata extraction
- exact factual grounding
- retrieval of small but important sections such as credits or faculty

### 10.2 Evaluation-Specific Questions

These questions focus on assessment and evaluation patterns of individual courses. They are more difficult than simple factual queries because the relevant information is often spread across a section or a table rather than a single metadata field.

Examples:

- What is the evaluation criteria for Generative AI and LLMs?
- What is the evaluation criteria for IoT Networks, Architectures and Applications?
- What is the evaluation criteria for Applied Machine Learning in Health Care?

These questions test:

- section-aware chunking
- semantic retrieval of evaluation blocks
- preservation of exact percentages and marks

### 10.3 Topic and Syllabus Questions

These questions test the ability of the system to retrieve descriptive course-content information rather than only structured metadata.

Examples:

- What topics are covered in IoT Networks, Architectures and Applications?
- What topics are covered in Cryptography?
- What will be taught in Theory of Computation?

These prompts are important because syllabus information tends to be distributed across larger chunks of text and requires stronger semantic matching.

### 10.4 Intermediate Comparison Questions

At the next level, two-course comparison questions were used. These require the system to retrieve information from multiple handouts and generate a comparative answer without mixing the course details.

Examples:

- Compare Cryptography and Network Security in terms of course focus, major topics, and security concepts.
- Compare the evaluation criteria of Generative AI and LLMs and Applied Machine Learning in Health Care.

These questions evaluate:

- multi-document retrieval
- cross-course comparison
- comparative summarization grounded in context

### 10.5 Complex Multi-Course Questions

The most advanced prompts used in the project were multi-course comparison questions involving three or more handouts. These were intended to test the full potential of the system and were stored in [Complex_Commands](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/Complex_Commands).

Examples:

- Compare the evaluation criteria of Generative AI and LLMs, IoT Networks, and Applied Machine Learning in Health Care. Mention the weightage of quizzes, assignments, projects, practical work, and end-sem evaluation for each course.
- Compare the topics covered in Generative AI and LLMs, Making Causal Inferences, and Applied Machine Learning in Health Care. Highlight the overlap and the distinct focus of each course.
- Across all uploaded handouts, which courses appear more theory-heavy and which appear more application-oriented? Use topics, credits, practical work, project components, and evaluation pattern to justify the comparison.

These complex prompts test:

- retrieval from multiple documents at once
- cross-course reasoning
- answer grounding across multiple sources
- the difference in answer style between the two compared LLMs

### 10.6 Importance of Using a Range of Question Types

This range of questions was important because it allowed the project to evaluate the RAG system more realistically. A system that performs well only on direct factual extraction may still fail on section-specific queries or cross-course comparisons. By using simple, intermediate, and complex prompts, the project was able to measure:

- exact factual answering ability
- section-aware retrieval quality
- topic-level semantic retrieval
- multi-document comparison ability
- robustness of different vector stores and embeddings
- differences between the two tested LLMs

Thus, the experimental design was not limited to one question style. It covered a progression from direct fact retrieval to complex comparative reasoning over the full uploaded dataset.

## 11. Experimental Analysis

A notebook was created to compare all major results:

- [results_comparison.ipynb](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/results_comparison.ipynb)

The notebook:

- loads all `ask` logs
- normalizes common question types
- builds side-by-side matrices for the major configurations
- loads `compare-models` logs
- exports summary CSVs for report usage

This notebook can be used to generate tables and screenshots for the final report.

For complex multi-course comparison runs, an additional notebook was created:

- [complex_results_comparison.ipynb](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/complex_results_comparison.ipynb)

This notebook focuses specifically on the large cross-course comparison questions and compares the outputs of the two hosted LLMs.

## 12. Limitations

The current system has the following limitations:

- some PDFs still contain noisy extracted text
- tables and broken formatting reduce retrieval quality
- model availability depends on the Hugging Face provider support attached to the token
- no formal automatic evaluation metric such as exact match or F1 was used

## 13. Conclusion

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
