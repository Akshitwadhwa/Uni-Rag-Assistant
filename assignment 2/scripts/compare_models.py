from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful e-commerce customer support assistant for ShopSphere. "
    "Answer clearly, politely, and only according to the store policy shown in the training data. "
    "Give short practical next steps and do not invent unsupported policies."
)


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


def generate(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a base-vs-fine-tuned comparison file for report examples.")
    parser.add_argument("--model-name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--test-file", default="data/ecommerce_support/test.jsonl")
    parser.add_argument("--output-file", default="artifacts/base_vs_finetuned.json")
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.test_file)[: args.max_samples]

    base_tokenizer, base_model = load_model(args.model_name, adapter_path=None)
    tuned_tokenizer, tuned_model = load_model(args.model_name, adapter_path=args.adapter_path)

    comparisons = []
    for row in rows:
        prompt = build_prompt(row["prompt"], args.system_prompt)
        base_answer = generate(base_tokenizer, base_model, prompt, args.max_new_tokens)
        tuned_answer = generate(tuned_tokenizer, tuned_model, prompt, args.max_new_tokens)
        comparisons.append(
            {
                "id": row["id"],
                "category": row["category"],
                "prompt": row["prompt"],
                "reference": row["response"],
                "base_model_answer": base_answer,
                "fine_tuned_answer": tuned_answer,
            }
        )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    print(f"Saved comparison file to {output_path}")


if __name__ == "__main__":
    main()
