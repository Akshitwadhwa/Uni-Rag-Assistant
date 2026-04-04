# Retrieval-Augmented Generation for Domain-Specific Question Answering

## 1. Objective

The objective of this assignment is to design and implement a Retrieval-Augmented Generation (RAG) system for domain-specific question answering. The developed system answers student queries by retrieving relevant information from uploaded course handouts and then using a Large Language Model (LLM) to generate a grounded response.

In this project, the selected domain is **course handout question answering**. Instead of asking a general-purpose LLM to answer academic queries from memory, the system first retrieves relevant content from uploaded course handouts and then uses that retrieved context to answer factual questions such as:

- How many credits are there in a course?
- What is the evaluation criteria of a subject?
- What topics are covered in a specific course?
- Who is the course faculty or coordinator?

This reduces hallucination and improves factual reliability.

## 2. Selected Domain

The implemented system is a **Personal Course Handout Assistant**. The knowledge base consists of real course handouts used in the current experiment:

- Generative AI and LLMs
- Cryptography
- Theory of Computation
- IoT Networks, Architectures and Applications
- Applied Machine Learning in Health Care

This domain was chosen because:

- course handouts are structured and factual documents
- students often need quick answers from them
- plain LLM prompting is unreliable for exact academic details
- RAG is particularly suitable when answers must be grounded in uploaded documents

## 3. Problem Statement

Large Language Models are strong at language generation but may provide incorrect or fabricated information when asked domain-specific factual questions. In the academic setting, this is problematic because students often ask questions requiring exact details such as credit structure, assessment scheme, attendance rules, or course content.

To address this issue, a RAG system was developed where:

1. course handouts are ingested into the system
2. the documents are preprocessed and split into chunks
3. embeddings are generated for the chunks
4. the chunks are stored in a vector database
5. a user query is embedded and matched against the most relevant chunks
6. the retrieved context is supplied to an LLM for final answer generation

The system therefore combines semantic retrieval and LLM generation to produce context-grounded answers.

## 4. System Architecture

The implemented pipeline contains the following stages:

1. Document ingestion
2. PDF text extraction and normalization
3. Metadata extraction
4. Section-aware chunking
5. Embedding generation
6. Vector database indexing
7. Query embedding and retrieval
8. Prompt construction with retrieved context
9. Answer generation using a hosted Hugging Face model

The source code is modular and organized as follows:

- [ingestion.py](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment/ingestion.py): document loading, PDF extraction, metadata extraction
- [chunking.py](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment/chunking.py): chunk creation and section-aware splitting
- [embeddings.py](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment/embeddings.py): embedding model support
- [vectorstores.py](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment/vectorstores.py): FAISS and Chroma implementations
- [prompting.py](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment/prompting.py): prompt construction
- [generation.py](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment/generation.py): hosted LLM backends
- [pipeline.py](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment/pipeline.py): end-to-end RAG pipeline
- [cli.py](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/src/rag_assignment/cli.py): command-line execution and output logging

## 5. Dataset and Knowledge Base

The knowledge base was created from uploaded course handouts stored in [data/course_handouts](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/data/course_handouts). The documents used in the experiment were:

- `Generative AI and LLMs_CourseHandout.pdf`
- `Cryptography_CSE3703.pdf`
- `Course Handout - ToC (Jan-June).pdf`
- `IoT Networks, Architectures and Applications-CSE 2023 Batch.pdf`
- `2025_course_handout_AMCH.pdf`

These documents contain important academic details such as:

- course code
- course title
- credits
- course faculty / coordinator
- evaluation pattern
- topics covered

### Preprocessing Performed

The system performs the following preprocessing operations:

- PDF text extraction using `pypdf`
- whitespace normalization
- cleanup of noisy PDF text patterns
- inline metadata extraction from irregular PDF headers
- extraction of structured fields like credits, course title, and course code
- extraction of evaluation summary for noisy PDFs

Since some PDFs had broken spacing and formatting, additional logic was added to improve extraction from noisy lines such as:

- `Course Code: CSE3720 Course Name: Generative AI and LLMs Credits: 3`
- `A ssessment Pattern`
- `A im of the Course`

This was necessary because raw PDF extraction often does not preserve layout properly.

## 6. Metadata Extraction

Before chunking, the system extracts structured metadata from each handout. The main fields are:

| Field | Description | Use |
|---|---|---|
| `course_code` | official code of the subject | exact subject matching |
| `subject_name` | course title | natural language queries |
| `credits` | number of credits | factual question answering |
| `ltp` | lecture-tutorial-practical pattern | academic details |
| `faculty` | instructor or coordinator | faculty-related queries |
| `semester` | semester or term | extra filtering |
| `evaluation_summary` | extracted assessment details | evaluation queries |
| `section_title` | local section name | section-aware retrieval |
| `section_type` | semantic section label | ranking support |

Metadata extraction improved the reliability of factual answers, especially for:

- credit queries
- faculty queries
- evaluation queries

The updated pipeline also creates synthetic chunks such as:

- `Structured Course Facts`
- `Credits`
- `Evaluation Summary`

These chunks improve retrieval for exact factual questions even when the original PDF headings are noisy.

## 7. Chunking Strategy

This project uses **section-aware chunking** as the default strategy. Instead of splitting the entire handout into arbitrary chunks only, the chunker attempts to preserve meaningful document units such as:

- course overview
- credits
- evaluation summary
- topics of the course
- faculty information

### Chunking Configurations Considered

| Strategy | Chunk Size | Overlap | Observation |
|---|---:|---:|---|
| Section-aware + fixed | 500 | 80 | Most effective for handouts |
| Fixed-only | 500 | 80 | Works, but can split key facts |
| Sentence-window | configurable | configurable | weaker on table-like PDF content |

### Why Section-Aware Chunking Worked Better

For this project, many questions were about exact fields such as credits and evaluation criteria. If these details are broken across unrelated chunks, retrieval becomes less precise. Section-aware chunking preserved these academic fields more effectively and made the retrieved context more interpretable.

## 8. Embedding Models

Two embedding models were used for comparison:

| Model | Source | Dimension | Role |
|---|---|---:|---|
| `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face | 384 | lightweight baseline |
| `BAAI/bge-base-en-v1.5` | Hugging Face | 768 | stronger semantic retrieval |

### Observations

- `all-MiniLM-L6-v2` was faster and lighter.
- `bge-base-en-v1.5` generally produced more relevant retrieval for complex semantic queries.
- For structured course-handout questions, `BGE` often retrieved cleaner support chunks for evaluation and syllabus questions.

## 9. Vector Database Comparison

Two vector databases were implemented:

| Vector DB | Description | Observation |
|---|---|---|
| FAISS | in-memory dense vector similarity search | fast and straightforward for local experiments |
| Chroma | persistent vector database with metadata support | convenient for repeated experiments and inspection |

### Practical Notes

- FAISS performed well for dense similarity retrieval.
- Chroma was easier to reuse during repeated experimental runs.
- Dimension mismatch handling had to be managed explicitly when switching embedding models.
- The system was updated so Chroma collections are namespaced per embedding model and FAISS checks embedding dimension compatibility more clearly.

## 10. Prompt Engineering

The prompt design instructs the LLM to:

- answer only from the retrieved course-handout context
- avoid unsupported claims
- preserve exact values such as marks, percentages, and credits
- give plain-English answers instead of markdown-heavy outputs
- naturally mention source files

This prompt design improved readability of logged answers and made the outputs more suitable for report inclusion.

## 11. LLM-Based Answer Generation

Answer generation was performed through the Hugging Face hosted inference API using:

- `meta-llama/Llama-3.1-8B-Instruct`

The codebase also supports hosted comparison runs through the `compare-models` command when a second supported provider model is available.

For the final retrieval comparison matrix, `Llama 3.1` was kept fixed so that vector database and embedding model comparisons remained fair.

## 12. Experimental Configurations

The main retrieval comparison matrix used in this project was:

1. `FAISS + MiniLM + Llama`
2. `FAISS + BGE + Llama`
3. `Chroma + MiniLM + Llama`
4. `Chroma + BGE + Llama`

These configurations were tested on a common question set:

- What is the evaluation criteria for IoT Networks, Architectures and Applications?
- What topics are covered in IoT Networks, Architectures and Applications?
- What is the evaluation criteria for Applied Machine Learning in Health Care?
- How many credits are there in Applied Machine Learning in Health Care?

Additional supporting runs were also performed for:

- Generative AI and LLMs
- Theory of Computation
- Cryptography
- faculty/coordinator queries

The generated logs were saved in:

- [FAISS_Outputs](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/FAISS_Outputs)
- [Chromo_Output](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/Chromo_Output)
- [Compare](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/Compare)

## 13. Sample Queries Used

The following representative queries were used during the experiments:

- What is the evaluation criteria for Generative AI and LLMs?
- How many credits are there in Theory of Computation?
- What topics are covered in Cryptography?
- What is the evaluation criteria for IoT Networks, Architectures and Applications?
- What topics are covered in IoT Networks, Architectures and Applications?
- What is the evaluation criteria for Applied Machine Learning in Health Care?
- How many credits are there in Applied Machine Learning in Health Care?
- Who is the instructor or coordinator for Applied Machine Learning in Health Care?

## 14. Results and Analysis

The comparative analysis notebook is available at [results_comparison.ipynb](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/results_comparison.ipynb). It loads the saved JSON logs and creates side-by-side result tables.

### Key Observations

- The system successfully answered factual questions on credits, evaluation criteria, topics, and faculty details.
- The retrieval stage had a strong effect on final answer quality.
- `BGE` generally provided stronger semantic retrieval than `MiniLM`, especially for descriptive questions such as topics and evaluation pattern.
- `MiniLM` still worked reasonably well and served as a strong baseline.
- `Chroma + BGE + Llama` produced one of the most complete and consistent result sets in the current experiments.
- `FAISS + BGE + Llama` was also strong and useful for demonstrating a second vector database.

### Example Analysis

For queries related to evaluation criteria:

- the system was able to retrieve the relevant evaluation chunks from the uploaded handouts
- the final response preserved weightages and percentages more reliably after prompt and extraction improvements

For credit-related queries:

- extracted metadata and synthetic `Credits` chunks significantly improved answer reliability

For topic-related queries:

- section-aware chunking helped preserve syllabus structure better than naive flat chunking

### Comparative Evaluation Summary

| Configuration | Retrieval Quality | Observed Strength |
|---|---|---|
| FAISS + MiniLM + Llama | good baseline | fast and simple |
| FAISS + BGE + Llama | strong | better semantic retrieval |
| Chroma + MiniLM + Llama | good | persistent and reusable |
| Chroma + BGE + Llama | strongest overall in current runs | good retrieval + convenient storage |

### Best Performing Setup

Based on the current experimental evidence, the best overall setup for this project was:

**Chroma + BGE + Llama**

Reasons:

- more consistent retrieval on shared IoT and healthcare question sets
- stronger semantic matching for structured academic queries
- convenient persistence and repeated querying through Chroma

If preferred for speed and simpler dense retrieval, `FAISS + BGE + Llama` can also be presented as a strong alternative.

## 15. Example Output Demonstration

Example outputs have been saved as JSON logs and can be shown in the report through screenshots. Recommended screenshots include:

- one successful answer for evaluation criteria
- one successful answer for credits
- one side-by-side comparison from the notebook
- one terminal run showing indexing

Suggested files to use for screenshots:

- any `ask_*.json` file from [FAISS_Outputs](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/FAISS_Outputs)
- any `ask_*.json` file from [Chromo_Output](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/Chromo_Output)
- one `compare_models_*.json` file from [Compare](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/Compare)

## 16. Limitations

This system has the following limitations:

- PDF text extraction is noisy for some documents
- table-heavy or badly formatted handouts reduce extraction quality
- some hosted Hugging Face models were not available under the current provider access
- retrieval quality still depends on chunk quality and metadata extraction
- no formal automatic evaluation metric such as exact match or F1 was used in this baseline

## 17. Future Work

Possible future improvements include:

- adding reranking with a cross-encoder
- using OCR or richer PDF parsers for difficult documents
- supporting hybrid retrieval with keyword + dense search
- adding a web interface for uploading handouts
- adding direct metadata-based answering for exact facts like credits and course code
- evaluating with quantitative metrics on a manually labelled query-answer set

## 18. Conclusion

This project successfully implemented a domain-specific RAG system for course-handout question answering. The system was able to ingest real academic handouts, extract metadata, chunk the documents, generate embeddings, store them in FAISS and Chroma, retrieve relevant context, and generate grounded answers using a hosted Hugging Face LLM.

The experiments showed that both embedding model selection and vector database choice significantly affect final performance. Among the tested configurations, `Chroma + BGE + Llama` emerged as the strongest setup in the current experiments, while `FAISS + BGE + Llama` also performed well as a competitive alternative.

Overall, the project demonstrates that RAG is an effective approach for reducing hallucination and improving factual answer quality in domain-specific academic question answering.
