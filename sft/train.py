import argparse

from datasets import load_from_disk
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from data import load_terminal_corpus


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT on Nemotron-Terminal-Corpus")

    # Model
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3.5-4B")
    p.add_argument("--output_dir", type=str, default="./output")

    # Data
    p.add_argument(
        "--subsets",
        nargs="+",
        default=None,
        help="Dataset subsets to use (default: all four)",
    )
    p.add_argument("--sample_frac", type=float, default=None, help="Sub-sample fraction per subset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--dataset_num_proc", type=int, default=8)
    p.add_argument(
        "--tokenized_dataset_path",
        type=str,
        default=None,
        help="If set, load a pre-tokenized dataset from this path (datasets.load_from_disk) "
        "instead of raw Nemotron-Terminal-Corpus.",
    )

    # Training
    p.add_argument("--num_gpus", type=int, default=8, help="Total GPU count (for grad accum calc)")
    p.add_argument("--max_length", type=int, default=32768)
    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--global_batch_size", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--logging_steps", type=float, default=0.01, help="<1 = ratio of total steps")
    p.add_argument("--save_steps", type=float, default=0.05, help="<1 = ratio of total steps")
    p.add_argument("--packing", action="store_true", default=False)

    return p.parse_args()


def main():
    args = parse_args()

    grad_accum = args.global_batch_size // (args.num_gpus * args.per_device_train_batch_size)

    if getattr(args, "tokenized_dataset_path", None):
        dataset = load_from_disk(args.tokenized_dataset_path)
    else:
        dataset = load_terminal_corpus(
            subsets=args.subsets,
            sample_frac=args.sample_frac,
            seed=args.seed,
            cache_dir=args.cache_dir,
        )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        max_length=args.max_length,
        bf16=True,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        report_to=["tensorboard", "wandb"],
        seed=args.seed,
        packing=args.packing,
        dataset_num_proc=args.dataset_num_proc,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    trainer = SFTTrainer(
        model=args.model_name_or_path,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
