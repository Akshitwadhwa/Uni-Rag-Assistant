# Assignment 2: Fine-Tuning an E-Commerce Support Assistant

This folder contains a custom synthetic dataset plus a full LoRA fine-tuning workflow for the task `e-commerce customer support assistant`.

## Dataset Summary

- Domain: e-commerce customer support
- Dataset style: prompt-response supervised fine-tuning
- Store name used in examples: `ShopSphere`
- Total examples: `720`
- Splits:
  - train: `576`
  - validation: `72`
  - test: `72`
- Categories: `18`

## Categories

- order tracking
- delayed delivery
- cancel before shipment
- cancel after shipment
- return request
- refund status
- exchange request
- wrong item received
- damaged item
- missing item
- payment failed
- payment deducted but no order created
- address change
- coupon not working
- warranty claim
- out of stock query
- account login issue
- invoice request

## Dataset Format

Each row contains:

- `id`
- `dataset_name`
- `domain`
- `store_name`
- `split`
- `category`
- `prompt`
- `response`

Example:

```json
{
  "id": "ecs_0001",
  "dataset_name": "ecommerce_customer_support_assistant",
  "domain": "e-commerce customer support",
  "store_name": "ShopSphere",
  "split": "train",
  "category": "refund_status",
  "prompt": "Hi, I returned order SS482193 and I want to know when my refund will come. Please help.",
  "response": "Refunds are processed after the returned item passes inspection. For prepaid orders, the amount usually reaches the original payment method within 5 to 7 business days, while COD refunds are sent to the selected bank account or UPI ID within 3 to 5 business days after approval."
}
```

## Policy Assumptions Used In Responses

The target replies are consistent with a fixed store policy:

- cancellations are allowed only before shipment
- eligible returns and exchanges are allowed within 7 days of delivery
- wrong, damaged, or missing item issues should be reported within 48 hours of delivery
- prepaid refunds usually take 5 to 7 business days after approval
- COD refunds usually take 3 to 5 business days after approval
- address changes are allowed only before shipment
- only one coupon can be used per order

## Files

- `data/ecommerce_support/all.jsonl`
- `data/ecommerce_support/train.jsonl`
- `data/ecommerce_support/validation.jsonl`
- `data/ecommerce_support/test.jsonl`
- `data/ecommerce_support/all.csv`
- `data/ecommerce_support/train.csv`
- `data/ecommerce_support/validation.csv`
- `data/ecommerce_support/test.csv`
- `data/ecommerce_support/stats.json`
- `scripts/generate_dataset.py`
- `scripts/finetune_lora.py`
- `scripts/infer.py`
- `scripts/evaluate.py`
- `scripts/compare_models.py`
- `requirements.txt`

## Regeneration

Run this from `assignment 2`:

```bash
python3 scripts/generate_dataset.py
```

## Fine-Tuning Setup

Recommended starter model:

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

Recommended training environment:

- Google Colab with GPU
- local machine only if you have a CUDA GPU

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

Example LoRA fine-tuning run:

```bash
python3 scripts/finetune_lora.py \
  --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train-file data/ecommerce_support/train.jsonl \
  --validation-file data/ecommerce_support/validation.jsonl \
  --output-dir artifacts/tinyllama-ecommerce-lora \
  --num-train-epochs 3 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-4
```

If your GPU supports mixed precision, add either:

```bash
--fp16
```

or:

```bash
--bf16
```

## Single-Prompt Inference

Base model:

```bash
python3 scripts/infer.py \
  --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "My order was delivered with a damaged item. What should I do?"
```

Fine-tuned model:

```bash
python3 scripts/infer.py \
  --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path artifacts/tinyllama-ecommerce-lora \
  --prompt "My order was delivered with a damaged item. What should I do?"
```

## Evaluate On Test Set

Base model:

```bash
python3 scripts/evaluate.py \
  --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --test-file data/ecommerce_support/test.jsonl \
  --output-file artifacts/base_eval.json
```

Fine-tuned model:

```bash
python3 scripts/evaluate.py \
  --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path artifacts/tinyllama-ecommerce-lora \
  --test-file data/ecommerce_support/test.jsonl \
  --output-file artifacts/finetuned_eval.json
```

## Create Report Examples

Generate side-by-side outputs for a small sample:

```bash
python3 scripts/compare_models.py \
  --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path artifacts/tinyllama-ecommerce-lora \
  --test-file data/ecommerce_support/test.jsonl \
  --output-file artifacts/base_vs_finetuned.json \
  --max-samples 12
```

## Suggested Report Flow

1. Problem statement and why fine-tuning is needed.
2. Domain choice: e-commerce customer support.
3. Dataset construction methodology and policy assumptions.
4. Dataset size, split, and category coverage.
5. Base model and why it was chosen.
6. Fine-tuning approach: LoRA with PEFT.
7. Training settings: epochs, batch size, learning rate, max sequence length.
8. Evaluation methodology: base vs fine-tuned comparison.
9. Example outputs before and after fine-tuning.
10. Limitations and future work.

## Notes

- The current scripts use a simple instruction format with a fixed system prompt.
- The evaluation script reports `exact_match` and `token_f1`, which are lightweight assignment-friendly metrics.
- Exact match will usually be strict for generation tasks, so sample output comparison matters in the report.
- If training on Colab, save the adapter directory from `artifacts/` so you can reuse it later.
