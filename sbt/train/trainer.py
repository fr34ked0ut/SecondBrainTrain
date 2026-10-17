from datetime import datetime
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from .dataset import load_tokenized


def run(cfg: dict, data_path: Path, device: str = "cuda") -> Path:
    t = cfg["training"]
    tokenizer = AutoTokenizer.from_pretrained(t["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        t["base_model"],
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=t["lora_r"],
            lora_alpha=t["lora_alpha"],
            lora_dropout=t["lora_dropout"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    splits = load_tokenized(data_path, tokenizer, t["max_seq_len"])
    run_name = datetime.now().strftime("sbt-%Y%m%d-%H%M")
    out = Path(t["output_dir"]) / run_name

    args = TrainingArguments(
        output_dir=str(out),
        run_name=run_name,
        num_train_epochs=t["epochs"],
        learning_rate=t["lr"],
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=t["grad_accum"],
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=3,
        bf16=device == "cuda",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )
    trainer.train()
    model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")

    latest = Path(t["output_dir"]) / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(out.resolve())
    return out