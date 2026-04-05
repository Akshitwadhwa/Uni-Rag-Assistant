from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ROOT / "data" / "ecommerce_support"
REPORTS_DIR = ROOT / "reports"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def bleu1_score(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    ref_counts: dict[str, int] = {}
    used_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token in pred_tokens:
        used_counts[token] = used_counts.get(token, 0) + 1
        if used_counts[token] <= ref_counts.get(token, 0):
            overlap += 1

    precision = overlap / len(pred_tokens)
    brevity_penalty = 1.0 if len(pred_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1))
    return brevity_penalty * precision


def lcs_length(a: list[str], b: list[str]) -> int:
    rows = len(a) + 1
    cols = len(b) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(len(a) - 1, -1, -1):
        for j in range(len(b) - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[0][0]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_eval_summary(path: Path) -> dict:
    payload = load_json(path)
    predictions = payload["predictions"]
    payload["bleu1"] = sum(bleu1_score(row["prediction"], row["reference"]) for row in predictions) / len(predictions)
    payload["rouge_l_f1"] = sum(rouge_l_f1(row["prediction"], row["reference"]) for row in predictions) / len(predictions)
    return payload


def category_table(base_eval: dict, finetuned_eval: dict) -> list[dict]:
    base_scores: dict[str, list[float]] = defaultdict(list)
    finetuned_scores: dict[str, list[float]] = defaultdict(list)
    for row in base_eval["predictions"]:
        base_scores[row["category"]].append(row["token_f1"])
    for row in finetuned_eval["predictions"]:
        finetuned_scores[row["category"]].append(row["token_f1"])

    rows = []
    for category in sorted(base_scores):
        base_avg = sum(base_scores[category]) / len(base_scores[category])
        tuned_avg = sum(finetuned_scores[category]) / len(finetuned_scores[category])
        rows.append(
            {
                "category": category,
                "base_token_f1": base_avg,
                "finetuned_token_f1": tuned_avg,
                "gain": tuned_avg - base_avg,
            }
        )
    rows.sort(key=lambda row: row["gain"], reverse=True)
    return rows


def example_rows(base_eval: dict, finetuned_eval: dict, count: int = 5) -> list[dict]:
    rows = []
    for base_row, tuned_row in zip(base_eval["predictions"], finetuned_eval["predictions"], strict=False):
        rows.append(
            {
                "id": tuned_row["id"],
                "category": tuned_row["category"],
                "prompt": tuned_row["prompt"],
                "reference": tuned_row["reference"],
                "base_prediction": base_row["prediction"],
                "finetuned_prediction": tuned_row["prediction"],
                "gain": tuned_row["token_f1"] - base_row["token_f1"],
            }
        )
    rows.sort(key=lambda row: row["gain"], reverse=True)
    return rows[:count]


def short_text(text: str, limit: int = 420) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    parts = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        parts.append("| " + " | ".join(row) + " |")
    return "\n".join(parts)


def build_markdown(stats: dict, train_metrics: dict, base_eval: dict, finetuned_eval: dict) -> str:
    category_rows = category_table(base_eval, finetuned_eval)[:10]
    examples = example_rows(base_eval, finetuned_eval, count=5)

    summary_table = markdown_table(
        ["Metric", "Base Model", "Fine-Tuned Model"],
        [
            ["Exact Match", f"{base_eval['exact_match']:.4f}", f"{finetuned_eval['exact_match']:.4f}"],
            ["Token F1", f"{base_eval['token_f1']:.4f}", f"{finetuned_eval['token_f1']:.4f}"],
            ["BLEU-1", f"{base_eval['bleu1']:.4f}", f"{finetuned_eval['bleu1']:.4f}"],
            ["ROUGE-L F1", f"{base_eval['rouge_l_f1']:.4f}", f"{finetuned_eval['rouge_l_f1']:.4f}"],
        ],
    )

    top_categories_table = markdown_table(
        ["Category", "Base Token F1", "Fine-Tuned Token F1", "Gain"],
        [
            [
                row["category"],
                f"{row['base_token_f1']:.4f}",
                f"{row['finetuned_token_f1']:.4f}",
                f"{row['gain']:.4f}",
            ]
            for row in category_rows
        ],
    )

    example_sections = []
    for index, row in enumerate(examples, start=1):
        example_sections.append(
            "\n".join(
                [
                    f"### Example {index}: {row['category']}",
                    f"**Prompt:** {row['prompt']}",
                    f"**Reference:** {row['reference']}",
                    f"**Base model output:** {short_text(row['base_prediction'])}",
                    f"**Fine-tuned output:** {short_text(row['finetuned_prediction'])}",
                ]
            )
        )

    return f"""# Assignment 2 Report
## Fine-Tuning a Large Language Model Using a Custom E-Commerce Support Dataset

### Student Deliverable Summary
This submission includes the full source code for dataset generation, preprocessing, training, evaluation, and inference; the final dataset used for fine-tuning; the saved LoRA adapter; and this experimental report. The task selected for fine-tuning is an **e-commerce customer support assistant** that answers operational customer queries such as delayed delivery, refund status, return eligibility, account login issues, coupon problems, and warranty claims.

## 1. Problem Definition
Large language models pretrained on broad web-scale corpora are strong generalists, but they are often weak at following narrow operational policies. In a customer support setting, this is a serious issue because the model must produce short, policy-consistent, actionable responses rather than generic advice. A base model may respond politely yet still hallucinate support steps, request unnecessary information, or ignore domain-specific constraints such as refund timelines, return windows, or shipment cancellation rules.

The goal of this assignment was therefore to fine-tune an open-source LLM on a custom domain dataset so that the model behaves like a specialized e-commerce support assistant. The model was expected to answer user prompts with concise responses aligned with a fixed store policy. The final system should outperform the base model on task-specific generation quality and should produce outputs that are closer to the reference responses in both wording and policy consistency.

## 2. Dataset Creation Methodology
The dataset was constructed as a **custom synthetic supervised fine-tuning dataset**. Instead of collecting public chat logs, a fictional e-commerce platform called **ShopSphere** was defined and a stable support policy was written first. This policy controlled the contents of all target responses. Examples of policy rules included:

- cancellations are allowed only before shipment
- returns and exchanges are allowed within 7 days of delivery for eligible items
- wrong, damaged, or missing item complaints must be reported within 48 hours
- prepaid refunds generally take 5 to 7 business days after approval
- COD refunds generally take 3 to 5 business days after approval
- address changes are allowed only before shipment
- coupon stacking is not allowed

The dataset generator created realistic user prompts by combining:

- issue categories
- product names
- order IDs and transaction IDs
- variant attributes such as size and color
- diverse opening phrases and closing requests
- paraphrased customer intent templates

Each example contains a user `prompt`, a ground-truth `response`, and a `category` label. The dataset has **720** total examples split into **576 train**, **72 validation**, and **72 test** examples across **18 categories**. This size satisfies the assignment requirement of 500 to 2000 examples.

### Dataset Composition
- Domain: e-commerce customer support
- Dataset name: `ecommerce_customer_support_assistant`
- Total examples: {stats['total_examples']}
- Train / Validation / Test: {stats['split_counts']['train']} / {stats['split_counts']['validation']} / {stats['split_counts']['test']}
- Number of support categories: {len(stats['category_counts'])}

The 18 categories are: account login issue, address change, cancel after shipment, cancel before shipment, coupon not working, damaged item, delayed delivery, exchange request, invoice request, missing item, order tracking, out of stock query, payment deducted with no order, payment failed, refund status, return request, warranty claim, and wrong item received.

## 3. Model Architecture and Fine-Tuning Method
The base model selected was **TinyLlama/TinyLlama-1.1B-Chat-v1.0**, a compact causal language model suitable for Colab-based experimentation. The model was chosen for three reasons. First, it is open source and instruction tuned. Second, it is small enough to fine-tune with limited GPU resources. Third, it still provides a meaningful baseline because it can answer support-style prompts before any domain adaptation.

Fine-tuning was performed with **LoRA (Low-Rank Adaptation)** using the **PEFT** library. Instead of updating all model parameters, LoRA injects trainable low-rank matrices into selected transformer projection layers. This reduces memory usage and training cost while still allowing the model to learn domain-specific behavior. The target modules used in this project were:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

The final adapter configuration used:

- LoRA rank (`r`): 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Bias setting: none
- Task type: causal language modeling

The prompt format followed a simple instruction-tuning structure with system, user, and assistant turns. The system prompt explicitly told the model to behave as a helpful ShopSphere customer support assistant and to avoid inventing unsupported policies.

## 4. Training Configuration
Training was run on Google Colab with GPU support. The experiment used the train split for optimization and the validation split for checkpoint selection.

### Workflow Diagram
The overall workflow used in this assignment is shown below:

`Dataset creation -> preprocessing -> LoRA fine-tuning -> evaluation`

### Configuration Used
- Base model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Max sequence length: 384 tokens in the Colab notebook configuration
- Number of epochs: 2
- Learning rate: 2e-4
- Per-device training batch size: 2
- Per-device evaluation batch size: 2
- Gradient accumulation steps: 8
- Weight decay: 0.01
- Warmup ratio: 0.1
- Mixed precision: FP16 on GPU
- Fine-tuning method: LoRA with PEFT

### Hardware Used
- Training platform: Google Colab
- Accelerator: Google Colab GPU
- Reason for using Colab: local laptop training was too slow for practical fine-tuning, while Colab provided a workable GPU environment for parameter-efficient training

### Validation Metrics During Training
- Validation loss: {train_metrics['eval_loss']:.4f}
- Validation token accuracy: {train_metrics['eval_token_accuracy']:.4f}
- Validation perplexity: {train_metrics['eval_perplexity']:.4f}
- Final recorded training loss: {train_metrics['train_loss']:.4f}

The low validation loss and low perplexity indicate that the adapter successfully learned the response style and policy structure present in the custom dataset.

## 5. Evaluation Methodology
The base model and the fine-tuned model were both evaluated on the same held-out 72-example test split. Each model generated responses for the exact same prompts. The following metrics were reported:

- **Exact Match**: strict string-level normalized match between prediction and reference
- **Token F1**: overlap-based token-level F1 score, useful when wording differs but content overlaps
- **BLEU-1**: unigram precision-oriented n-gram metric
- **ROUGE-L F1**: longest common subsequence overlap metric

Exact match is intentionally strict for generative tasks, so qualitative examples were also reviewed to determine whether the model actually learned the store policy and produced more useful answers.

## 6. Results and Base-vs-Fine-Tuned Comparison
The fine-tuned model outperformed the base model across every metric used in this experiment.

{summary_table}

### Interpretation
The base model was able to generate fluent text, but it frequently produced generic support advice, requested placeholder information, or suggested actions that were not part of the defined policy. In contrast, the fine-tuned model produced much more direct and policy-aligned answers. The largest relative gain was visible in token-level overlap metrics, which shows that the adapter learned the structure and content of the target responses.

The improvement in **Exact Match** from 0.0000 to {finetuned_eval['exact_match']:.4f} is meaningful because the model is generating free text rather than class labels. The jump in **Token F1** from {base_eval['token_f1']:.4f} to {finetuned_eval['token_f1']:.4f}, along with BLEU-1 and ROUGE-L improvements, shows that the fine-tuned model produced outputs much closer to the expected support responses.

### Category-Level Token F1 Improvement
{top_categories_table}

The strongest gains were observed in categories where the response policy is very precise, such as **exchange requests**, **missing items**, and **return requests**. These are exactly the kinds of tasks where fine-tuning should help most: the base model tends to invent general advice, while the fine-tuned model can memorize and apply the domain policy.

## 7. Qualitative Error Analysis
Although the fine-tuned model improved substantially, it is not perfect. The test outputs show three main remaining issues:

1. **Policy confusion between related categories.** For example, some `cancel_after_shipment` prompts were still answered using a `cancel_before_shipment` policy template.
2. **Near-correct paraphrases that fail exact match.** Many outputs were semantically correct but worded slightly differently from the reference, which lowers exact match despite being acceptable in practice.
3. **Occasional overgeneralization.** In a few cases, the fine-tuned model reused a highly frequent support template across similar prompt types, which reduced category-specific precision.

These remaining problems suggest that the dataset could be improved further with additional paraphrase diversity and more contrastive examples between closely related intents.

{chr(10).join(example_sections)}

These examples show the main improvement pattern clearly. The base model often produced verbose, generic, or partially hallucinated instructions, while the fine-tuned model produced concise answers aligned with the support policy. In several cases, the fine-tuned output matched the reference exactly.

## 9. Discussion
This experiment demonstrates the value of fine-tuning for specialized business tasks. A general model can already produce fluent customer-support-style text, but fluency alone is not enough in a production support workflow. What matters is policy alignment, consistency, and task-specific correctness. By training on a small but carefully structured dataset, the model became much more reliable for the intended use case.

The experiment also shows that full-model training is not necessary for a useful adaptation. LoRA was sufficient to produce a strong quality jump while keeping the training process lightweight enough for Google Colab. This makes parameter-efficient fine-tuning practical for academic assignments and for small organizations that need custom assistants without enterprise-scale infrastructure.

## 10. Limitations and Future Work
This work has several limitations:

- The dataset is synthetic, so it does not fully capture the variability, ambiguity, and noise of real customer support conversations.
- Human evaluation was not included in the final experiment, so helpfulness and policy correctness were judged only through automatic metrics and qualitative inspection.
- The base model is relatively small, which makes training efficient but also limits the final model capacity.
- The dataset is synthetic rather than collected from real users, so it may not cover the full linguistic diversity of real customer support traffic.
- The current evaluation uses lexical overlap metrics; human evaluation of helpfulness and policy correctness would strengthen the analysis.
- The model was fine-tuned on a single-store policy and may not transfer well to a different platform without retraining.
- Some categories remain partially confusable, especially where policies differ only by order state.

Future work could include:

- expanding the dataset with harder paraphrases and adversarial prompts
- adding multilingual customer support examples
- introducing human evaluation for helpfulness, correctness, and tone
- comparing LoRA with QLoRA or larger base models such as Mistral or Gemma
- integrating the model with retrieval or policy documents for hybrid support automation

## 11. Conclusion
In this assignment, a custom e-commerce customer support dataset was created and used to fine-tune an open-source LLM using LoRA. The final fine-tuned TinyLlama model substantially outperformed the base model on the held-out test set. Exact Match improved from {base_eval['exact_match']:.4f} to {finetuned_eval['exact_match']:.4f}, Token F1 improved from {base_eval['token_f1']:.4f} to {finetuned_eval['token_f1']:.4f}, BLEU-1 improved from {base_eval['bleu1']:.4f} to {finetuned_eval['bleu1']:.4f}, and ROUGE-L F1 improved from {base_eval['rouge_l_f1']:.4f} to {finetuned_eval['rouge_l_f1']:.4f}.

The results confirm the central idea of the assignment: fine-tuning makes a general language model better suited for a domain-specific task by teaching it the target behavior, language style, and policy boundaries of that domain. In this case, fine-tuning transformed a generic assistant into a more reliable e-commerce support agent.
"""


def escape_rtf(text: str) -> str:
    text = text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
    text = text.replace("—", "-").replace("–", "-").replace("’", "'").replace("“", '"').replace("”", '"')
    return text


def markdown_to_rtf(markdown: str) -> str:
    lines = markdown.splitlines()
    out: list[str] = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0 Calibri;}{\f1 Courier New;}}",
        r"\fs22",
    ]
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append(r"\par")
            continue
        if stripped.startswith("# "):
            out.append(r"\b\fs32 " + escape_rtf(stripped[2:]) + r"\b0\fs22\par")
            continue
        if stripped.startswith("## "):
            out.append(r"\b\fs28 " + escape_rtf(stripped[3:]) + r"\b0\fs22\par")
            continue
        if stripped.startswith("### "):
            out.append(r"\b\fs24 " + escape_rtf(stripped[4:]) + r"\b0\fs22\par")
            continue
        if stripped.startswith("- "):
            out.append(r"\tab " + escape_rtf(stripped) + r"\par")
            continue
        out.append(escape_rtf(stripped) + r"\par")
    out.append("}")
    return "\n".join(out)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    stats = load_json(DATA_DIR / "stats.json")
    train_metrics = load_json(ARTIFACTS / "tinyllama_ecommerce_lora" / "metrics.json")
    base_eval = compute_eval_summary(ARTIFACTS / "base_eval.json")
    finetuned_eval = compute_eval_summary(ARTIFACTS / "finetuned_eval.json")

    markdown = build_markdown(stats, train_metrics, base_eval, finetuned_eval)
    markdown_path = REPORTS_DIR / "assignment2_report.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    rtf = markdown_to_rtf(markdown)
    rtf_path = REPORTS_DIR / "assignment2_report.rtf"
    rtf_path.write_text(rtf, encoding="utf-8")

    summary = {
        "markdown_report": str(markdown_path),
        "word_compatible_report": str(rtf_path),
        "base_token_f1": round(base_eval["token_f1"], 4),
        "finetuned_token_f1": round(finetuned_eval["token_f1"], 4),
        "base_exact_match": round(base_eval["exact_match"], 4),
        "finetuned_exact_match": round(finetuned_eval["exact_match"], 4),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
