from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful e-commerce customer support assistant for ShopSphere. "
    "Answer clearly, politely, and only according to the store policy shown in the training data. "
    "Give short practical next steps and do not invent unsupported policies."
)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, ref_counts.get(token, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def bleu1(prediction: str, reference: str) -> float:
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


def lcs_length(left: list[str], right: list[str]) -> int:
    dp = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i in range(len(left) - 1, -1, -1):
        for j in range(len(right) - 1, -1, -1):
            if left[i] == right[j]:
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


def build_prompt(user_prompt: str, system_prompt: str) -> str:
    return (
        f"<|system|>\n{system_prompt}\n"
        f"<|user|>\n{user_prompt.strip()}\n"
        f"<|assistant|>\n"
    )


def load_rows(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_model(model_name: str, adapter_path: str | None):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def generate_answer(tokenizer, model, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the base or fine-tuned model on the e-commerce support test set.")
    parser.add_argument("--model-name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--test-file", default="data/ecommerce_support/test.jsonl")
    parser.add_argument("--output-file", default="artifacts/eval_predictions.json")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.test_file)
    if args.limit is not None:
        rows = rows[: args.limit]

    tokenizer, model = load_model(args.model_name, args.adapter_path)

    predictions = []
    exact_scores = []
    f1_scores = []
    bleu1_scores = []
    rouge_l_scores = []

    for row in tqdm(rows, desc="Evaluating"):
        prompt = build_prompt(row["prompt"], args.system_prompt)
        prediction = generate_answer(
            tokenizer,
            model,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        em = exact_match(prediction, row["response"])
        f1 = token_f1(prediction, row["response"])
        bleu = bleu1(prediction, row["response"])
        rouge = rouge_l_f1(prediction, row["response"])
        exact_scores.append(em)
        f1_scores.append(f1)
        bleu1_scores.append(bleu)
        rouge_l_scores.append(rouge)
        predictions.append(
            {
                "id": row["id"],
                "category": row["category"],
                "prompt": row["prompt"],
                "reference": row["response"],
                "prediction": prediction,
                "exact_match": em,
                "token_f1": f1,
                "bleu1": bleu,
                "rouge_l_f1": rouge,
            }
        )

    summary = {
        "model_name": args.model_name,
        "adapter_path": args.adapter_path,
        "examples_evaluated": len(predictions),
        "exact_match": sum(exact_scores) / len(exact_scores) if exact_scores else 0.0,
        "token_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "bleu1": sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0.0,
        "rouge_l_f1": sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0,
        "predictions": predictions,
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
