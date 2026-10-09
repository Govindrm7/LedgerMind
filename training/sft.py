"""Stage 1, SFT: teach the output contract (evidence with pointers, a plan, no number).

Usage::

    uv run python training/sft.py --config training/configs/sft_qwen3_4b.yaml
"""

from __future__ import annotations

import argparse

from common import (
    fits,
    install_cpu_kernel_fallback,
    load_config,
    load_tokenizer,
    write_run_metadata,
)
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from ledgermind.data.prepare import read_jsonl
from ledgermind.training.records import sft_record


def build_dataset(path: str, tokenizer, max_length: int, limit: int | None) -> tuple[Dataset, int]:
    records = [sft_record(ex) for ex in read_jsonl(path)]
    kept = [r for r in records if fits(tokenizer, r["prompt"], r["completion"], max_length)]
    dropped = len(records) - len(kept)
    if limit:
        kept = kept[:limit]
    return Dataset.from_list(kept), dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if install_cpu_kernel_fallback():
        print("Triton unavailable: using the PyTorch log prob fallback (smoke tests only)")

    tokenizer = load_tokenizer(cfg["model"])
    train, dropped = build_dataset(
        cfg["train_file"], tokenizer, cfg["max_length"], cfg.get("max_train_examples")
    )
    evaluation, _ = build_dataset(
        cfg["eval_file"], tokenizer, cfg["max_length"], cfg.get("max_eval_examples")
    )
    print(f"train {len(train)} (dropped {dropped} over max_length), eval {len(evaluation)}")

    lora = cfg["lora"]
    peft_config = LoraConfig(
        r=lora["rank"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora.get("dropout", 0.0),
        target_modules=lora.get("target_modules", "all-linear"),
        task_type="CAUSAL_LM",
    )
    t = cfg["train"]
    sft_args = SFTConfig(
        output_dir=cfg["output_dir"],
        max_length=cfg["max_length"],
        completion_only_loss=True,
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_steps=t.get("warmup", 0.05),
        num_train_epochs=t.get("epochs", 2),
        max_steps=t.get("max_steps", -1),
        per_device_train_batch_size=t["per_device_batch_size"],
        per_device_eval_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        bf16=t.get("bf16", True),
        logging_steps=t.get("logging_steps", 10),
        eval_strategy="steps" if t.get("eval_steps") else "no",
        eval_steps=t.get("eval_steps"),
        save_strategy=t.get("save_strategy", "epoch"),
        report_to=t.get("report_to", "none"),
        seed=t.get("seed", 0),
        model_init_kwargs=cfg.get("model_init_kwargs"),
        use_cpu=t.get("use_cpu", False),
    )
    trainer = SFTTrainer(
        model=cfg["model"],
        args=sft_args,
        train_dataset=train,
        eval_dataset=evaluation if len(evaluation) else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    sample = trainer.train_dataset[0]
    if sample["input_ids"][-1] != tokenizer.eos_token_id:
        raise RuntimeError("completions must end with EOS so the model learns to stop")
    result = trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    write_run_metadata(
        cfg["output_dir"],
        cfg,
        train_examples=len(train),
        dropped=dropped,
        train_loss=result.training_loss,
        log_history=trainer.state.log_history,
    )


if __name__ == "__main__":
    main()
