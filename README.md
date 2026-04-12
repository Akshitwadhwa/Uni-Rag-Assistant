# Gen AI Group Assignment

This repository contains two separate assignments completed for the Generative AI and LLMs course:

- `assignment 1`: Retrieval-Augmented Generation (RAG) for course handout question answering
- `assignment 2`: LoRA fine-tuning for an e-commerce support assistant

Both assignments are organized independently, with their own code, data, reports, and experiment artifacts.

## Repository Structure

```text
Gen_AI group assignment/
├── assignment 1/
│   ├── src/
│   ├── data/
│   ├── artifacts/
│   ├── output/
│   ├── report.md
│   ├── report_template.md
│   ├── commands.md
│   ├── results_comparison.ipynb
│   └── complex_results_comparison.ipynb
├── assignment 2/
│   ├── scripts/
│   ├── data/
│   ├── models/
│   ├── outputs/
│   └── README.md
└── README.md
```

## Assignment 1

**Title:** Retrieval-Augmented Generation for Domain-Specific Question Answering

This assignment implements a RAG pipeline for answering questions from uploaded course handouts. The system ingests PDF handouts, extracts metadata, chunks the content, creates embeddings, stores them in a vector database, retrieves relevant chunks for a query, and generates grounded answers with an LLM.

### Assignment 1 Domain

The chosen domain is academic course handouts. The knowledge base includes handouts such as:

- Generative AI and LLMs
- Cryptography
- Theory of Computation
- IoT Networks, Architectures and Applications
- Applied Machine Learning in Health Care
- Making Causal Inferences
- Network Security

### Assignment 1 Features

- PDF document ingestion and preprocessing
- metadata extraction for course title, course code, credits, faculty, and evaluation summary
- section-aware chunking
- multiple embedding models
- two vector databases: `FAISS` and `Chroma`
- Hugging Face hosted LLM inference
- model comparison experiments
- saved experiment logs and notebooks for result analysis

### Main Files for Assignment 1

- [Assignment 1 README](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/README.md)
- [Assignment 1 Report](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/report.md)
- [Assignment 1 Commands](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/commands.md)
- [Assignment 1 Results Notebook](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/results_comparison.ipynb)
- [Assignment 1 Complex Results Notebook](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/complex_results_comparison.ipynb)

### Assignment 1 Core Configurations Compared

- `FAISS + MiniLM + Llama`
- `FAISS + BGE + Llama`
- `Chroma + MiniLM + Llama`
- `Chroma + BGE + Llama`

### Running Assignment 1

```bash
cd "/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1"
source ../.venv/bin/activate
```

Then refer to [commands.md](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 1/commands.md) for demo and experiment commands.

## Assignment 2

**Title:** Fine-Tuning an E-Commerce Support Assistant

This assignment focuses on parameter-efficient fine-tuning of a small language model using LoRA for customer support dialogue generation in the e-commerce domain.

### Assignment 2 Objective

The goal is to adapt a base model to respond more effectively to e-commerce support queries such as:

- order status
- refunds
- returns
- payment failures
- account issues
- shipping and delivery questions

### Assignment 2 Dataset

The project includes a synthetic e-commerce support dataset with:

- total examples: `720`
- training examples: `576`
- validation examples: `72`
- test examples: `72`
- support categories: `18`

### Assignment 2 Features

- dataset generation and preparation
- train, validation, and test splits
- LoRA fine-tuning pipeline
- inference script
- evaluation script
- model comparison workflow

### Main Files for Assignment 2

- [Assignment 2 README](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 2/README.md)
- [Dataset Generator](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 2/scripts/generate_dataset.py)
- [Fine-Tuning Script](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 2/scripts/finetune_lora.py)
- [Inference Script](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 2/scripts/infer.py)
- [Evaluation Script](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 2/scripts/evaluate.py)
- [Model Comparison Script](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 2/scripts/compare_models.py)

### Running Assignment 2

```bash
cd "/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 2"
source ../.venv/bin/activate
```

Then follow the detailed instructions in [assignment 2/README.md](/Users/Lenovo/Desktop/sem 6/Gen_AI group assignment/assignment 2/README.md).

## Environment Notes

- A shared virtual environment is present at the repository root: `.venv`
- Assignment 1 relies on Hugging Face hosted inference for answer generation
- Assignment 2 fine-tuning is better suited to GPU or Colab for efficient training


## Summary

This repository demonstrates two core Generative AI workflows:

- retrieval-augmented question answering over domain-specific documents
- parameter-efficient fine-tuning for domain-specific conversational support

Together, the two assignments cover both major practical directions in applied GenAI systems: grounding with retrieval and specialization with fine-tuning.
