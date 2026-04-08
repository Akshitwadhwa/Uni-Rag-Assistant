from __future__ import annotations

import argparse
import inspect
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful e-commerce customer support assistant for ShopSphere. "
    "Answer clearly, politely, and only according to the store policy shown in the training data. "
    "Give short practical next steps and do not invent unsupported policies."
)


@dataclass
class DatasetPaths:
    train: str
    validation: str
    test: str | None = None


def build_prompt(user_prompt: str, system_prompt: str) -> str:
    return (
        f"<|system|>\n{system_prompt}\n"
        f"<|user|>\n{user_prompt.strip()}\n"
        f"<|assistant|>\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM with LoRA on the e-commerce support dataset.")
    parser.add_argument("--model-name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--train-file", default="data/ecommerce_support/train.jsonl")
    parser.add_argument("--validation-file", default="data/ecommerce_support/validation.jsonl")
    parser.add_argument("--output-dir", default="artifacts/tinyllama-ecommerce-lora")
    parser.add_argument("--max-length", type=int, default=374)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(model_name: str, use_gradient_checkpointing: bool):
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model.config.use_cache = not use_gradient_checkpointing
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def build_lora_model(model, args: argparse.Namespace):
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def tokenize_example(example: dict, tokenizer, max_length: int, system_prompt: str) -> dict:
    prompt_text = build_prompt(example["prompt"], system_prompt)
    response_text = example["response"].strip() + tokenizer.eos_token

    prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)
    response_tokens = tokenizer(response_text, add_special_tokens=False)

    input_ids = prompt_tokens["input_ids"] + response_tokens["input_ids"]
    attention_mask = prompt_tokens["attention_mask"] + response_tokens["attention_mask"]
    labels = [-100] * len(prompt_tokens["input_ids"]) + response_tokens["input_ids"]

    input_ids = input_ids[:max_length]
    attention_mask = attention_mask[:max_length]
    labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_datasets(paths: DatasetPaths, tokenizer, max_length: int, system_prompt: str):
    dataset = load_dataset(
        "json",
        data_files={
            "train": paths.train,
            "validation": paths.validation,
        },
    )

    def mapper(example: dict) -> dict:
        return tokenize_example(example, tokenizer, max_length=max_length, system_prompt=system_prompt)

    tokenized = dataset.map(mapper, remove_columns=dataset["train"].column_names)
    return tokenized


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = torch.tensor(logits).argmax(dim=-1)
    labels = torch.tensor(labels)

    shifted_predictions = predictions[:, :-1].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    valid_mask = shifted_labels != -100

    if valid_mask.sum().item() == 0:
        return {"token_accuracy": 0.0}

    correct = ((shifted_predictions == shifted_labels) & valid_mask).sum().item()
    total = valid_mask.sum().item()
    return {"token_accuracy": correct / total}


def save_run_config(args: argparse.Namespace, output_dir: Path) -> None:
    payload = vars(args).copy()
    payload["output_dir"] = str(output_dir)
    (output_dir / "run_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(args, output_dir)

    tokenizer = load_tokenizer(args.model_name)
    model = load_model(args.model_name, use_gradient_checkpointing=args.gradient_checkpointing)
    model = build_lora_model(model, args)

    dataset_paths = DatasetPaths(
        train=args.train_file,
        validation=args.validation_file,
    )
    tokenized = build_datasets(
        dataset_paths,
        tokenizer=tokenizer,
        max_length=args.max_length,
        system_prompt=args.system_prompt,
    )

    training_kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "remove_unused_columns": False,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "seed": args.seed,
    }
    training_args_signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in training_args_signature.parameters:
        training_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in training_args_signature.parameters:
        training_kwargs["eval_strategy"] = "steps"

    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "data_collator": DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        "compute_metrics": compute_metrics,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    train_result = trainer.train()
    metrics = trainer.evaluate()
    metrics["train_loss"] = train_result.training_loss
    if "eval_loss" in metrics:
        metrics["eval_perplexity"] = math.exp(metrics["eval_loss"])

    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
