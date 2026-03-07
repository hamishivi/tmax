import argparse
import os

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from data import load_terminal_corpus


class SFTTrainerSP(SFTTrainer):
    """Fixes a bug in transformers' deepspeed_sp_compute_loss that accesses
    accelerator.torch_device_mesh['sp'] which is None for DeepSpeed SP
    (accelerate's build_device_mesh explicitly returns None for it).

    This override uses deepspeed.utils.groups instead of device_mesh.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        pc = getattr(self.accelerator, "parallelism_config", None)
        if pc is not None and pc.sp_backend == "deepspeed" and pc.sp_enabled and model.training:
            if "labels" not in inputs and "shift_labels" in inputs:
                inputs["labels"] = inputs["shift_labels"]

            outputs = model(**inputs)
            loss = outputs.loss

            from deepspeed.utils import groups

            sp_group = groups._get_sequence_parallel_group()
            sp_world_size = pc.sp_size

            losses_per_rank = torch.distributed.nn.functional.all_gather(loss, group=sp_group)
            good_tokens = (inputs["shift_labels"] != -100).view(-1).sum()
            good_tokens_per_rank = torch.distributed.nn.functional.all_gather(good_tokens, group=sp_group)

            total_loss = sum(
                losses_per_rank[rank] * good_tokens_per_rank[rank]
                for rank in range(sp_world_size)
                if good_tokens_per_rank[rank] > 0
            )
            total_good_tokens = sum(good_tokens_per_rank)
            loss = total_loss / max(total_good_tokens, 1)

            return (loss, outputs) if return_outputs else loss

        return super().compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )


class CollatorWithPositionIds:
    """Wraps a collator to inject global position_ids into the batch.

    Ulysses SP needs position_ids in the batch so the dataloader adapter can
    shard them per-rank, giving each rank the correct rotary-embedding positions.
    Without this, every rank would generate local [0..chunk_len) positions.
    """

    def __init__(self, inner_collator):
        self.inner = inner_collator

    def __call__(self, features):
        batch = self.inner(features)
        if "position_ids" not in batch and "input_ids" in batch:
            seq_len = batch["input_ids"].shape[1]
            batch["position_ids"] = (
                torch.arange(seq_len)
                .unsqueeze(0)
                .expand(batch["input_ids"].shape[0], -1)
                .contiguous()
            )
        return batch


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
        nargs="+",
        default=None,
        help="If set, load pre-tokenized dataset(s) from these paths. "
        "Multiple paths will be concatenated.",
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

    # Detect SP from accelerate's parallelism_config env vars (set by YAML)
    sp_size = int(os.environ.get("PARALLELISM_CONFIG_SP_SIZE", "1"))
    sp_enabled = sp_size > 1

    dp_world_size = max(1, args.num_gpus // sp_size)
    grad_accum = args.global_batch_size // (dp_world_size * args.per_device_train_batch_size)

    if sp_enabled:
        # Fix: accelerate's deepspeed_ulysses_dl_adapter and deepspeed_sp_compute_loss
        # use deepspeed.utils.groups to look up SP process groups, but nobody populates
        # groups.mpu. Point it at the module that register_with_transformers will
        # initialize (the same object it returns as mpu).
        import deepspeed.runtime.sequence_parallel.parallel_state_sp as sp_state
        import deepspeed.utils.groups as ds_groups

        ds_groups.mpu = sp_state

        # Fix two issues with position_ids in UlyssesSPAttentionHF.forward:
        # 1. all_gather requires contiguous tensors but gradient checkpointing creates views.
        # 2. Ulysses hardcodes dim=1 for gathering position_ids, but MROPE models (Qwen3.5)
        #    produce 3D position_ids [rope_heads, batch, seq] where the seq dim is 2, not 1.
        #    Since rotary embeddings are already applied before attention, position_ids at
        #    this stage is only used by FA2 for packed-sequence detection. For 3D (MROPE)
        #    position_ids we drop it; for 2D we just ensure contiguity.
        from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF

        _orig_forward = UlyssesSPAttentionHF.forward

        def _patched_forward(self, module, query, key, value, attention_mask, *args, **kwargs):
            if "position_ids" in kwargs and isinstance(kwargs["position_ids"], torch.Tensor):
                pos = kwargs["position_ids"]
                if pos.ndim > 2:
                    kwargs.pop("position_ids")
                elif not pos.is_contiguous():
                    kwargs["position_ids"] = pos.contiguous()
            return _orig_forward(self, module, query, key, value, attention_mask, *args, **kwargs)

        UlyssesSPAttentionHF.forward = _patched_forward

    # Load dataset
    if getattr(args, "tokenized_dataset_path", None):
        if isinstance(args.tokenized_dataset_path, list) and len(args.tokenized_dataset_path) > 1:
            from datasets import concatenate_datasets
            dataset = concatenate_datasets([load_from_disk(p) for p in args.tokenized_dataset_path])
        else:
            path = args.tokenized_dataset_path[0] if isinstance(args.tokenized_dataset_path, list) else args.tokenized_dataset_path
            dataset = load_from_disk(path)
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
        fp16=False,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        report_to=["tensorboard", "wandb"],
        seed=args.seed,
        packing=args.packing,
        dataset_num_proc=args.dataset_num_proc,
        pad_to_multiple_of=sp_size if sp_enabled else None,
        model_init_kwargs={
            "attn_implementation": "flash_attention_2",
            "torch_dtype": "bfloat16",
        } if sp_enabled else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    trainer_cls = SFTTrainerSP if sp_enabled else SFTTrainer
    trainer = trainer_cls(
        model=args.model_name_or_path,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    if sp_enabled:
        # Qwen3.5 stores model params inside text_config; expose them at the top
        # level so DeepSpeed's register_with_transformers can find them.
        cfg = trainer.model.config
        if hasattr(cfg, "text_config"):
            tc = cfg.text_config
            for attr in ["num_attention_heads", "num_key_value_heads", "num_hidden_layers", "hidden_size"]:
                if not hasattr(cfg, attr) and hasattr(tc, attr):
                    setattr(cfg, attr, getattr(tc, attr))
            if not hasattr(cfg, "head_dim"):
                setattr(cfg, "head_dim", getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads))

        trainer.data_collator = CollatorWithPositionIds(trainer.data_collator)

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
