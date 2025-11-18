import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Any

import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
import datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune nucleotide MLM with custom LR schedule.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset folder (expects train.txt / valid.txt or a HF saved dataset).")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="/data2/zhanghaohong/transformers_model/nucleotide-transformer-v2-50m-multi-species",)
    parser.add_argument("--max_length", type=int, default=1000, help="Sequence length (tokens).")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--lr_init", type=float, default=5e-5, help="Initial LR at step 0.")
    parser.add_argument("--lr_peak", type=float, default=1e-4, help="Peak LR after warmup.")
    parser.add_argument("--warmup_steps", type=int, default=16000, help="Linear warmup steps.")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--devices", type=str, default="0,1,2,3,4,5,6,7", help="Comma-separated GPU ids to use")
    return parser.parse_args()


def load_local_text_dataset(path: str) -> datasets.DatasetDict:
    # Try HF saved dataset first
    return datasets.load_from_disk(path)


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Set visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    max_length = args.max_length + 1 # account for special tokens

    raw_datasets = load_local_text_dataset(args.dataset_path)

    def tokenize_fn(batch):
        return tokenizer(
            batch["Sequence"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = raw_datasets.map(
        tokenize_fn,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=[c for c in raw_datasets["train"].column_names if c != "Sequence"],
    )

    # Rename input_ids already produced by tokenizer; ensure labels for MLM come from data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    # Estimate total training steps (Trainer will refine). Use len(train_dataset) / batch_size_effective per epoch.
    train_dataset = tokenized["train"]
    steps_per_epoch = math.ceil(
        len(train_dataset) / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    )
    total_estimated_steps = steps_per_epoch * args.num_train_epochs

    effective_tokens_per_step = (
        args.per_device_train_batch_size * max_length * args.gradient_accumulation_steps
    )
    print(f"Estimated steps: {total_estimated_steps}")
    print(f"Effective tokens per optimization step (approx): {effective_tokens_per_step}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps" if "validation" in tokenized else "no",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        learning_rate=args.lr_peak,  # Peak LR; scheduler scales it.
        warmup_steps=0,  # We implement our own warmup.
        weight_decay=args.weight_decay,
        report_to=[],
        push_to_hub=args.push_to_hub,
        fp16=torch.cuda.is_available(),
    )

    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=tokenized.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Create optimizer & default scheduler then override with custom LambdaLR
    trainer.create_optimizer_and_scheduler(num_training_steps=total_estimated_steps)

    warmup_steps = args.warmup_steps
    lr_init = args.lr_init
    lr_peak = args.lr_peak

    def lr_lambda(step: int):
        # step starts at 0
        if step < warmup_steps:
            current_lr = lr_init + (lr_peak - lr_init) * (step / warmup_steps)
        else:
            # sqrt decay: lr = lr_peak * sqrt(warmup_steps / step)
            # avoid division by zero by ensuring step >= 1
            s = max(step, 1)
            current_lr = lr_peak * math.sqrt(warmup_steps / s)
        return current_lr / lr_peak

    trainer.lr_scheduler = LambdaLR(trainer.optimizer, lr_lambda=lr_lambda)

    print("Starting training with custom warmup + sqrt decay schedule.")
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()