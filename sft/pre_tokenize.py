import argparse
import json
from typing import Any

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

from data import load_converted_corpus

from trl.data_utils import maybe_convert_to_chatml, truncate_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-tokenize Nemotron-Terminal-Corpus for SFT")

    # Model / tokenizer
    p.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Model name or path whose tokenizer to use (no model weights are loaded).",
    )

    # Data loading
    p.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing converted Parquet files (default: preprocessing/terminus2_sweagent).",
    )
    p.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Source labels to include (e.g. 'nvidia/Nemotron-Terminal-Corpus/skill_based_easy').",
    )
    p.add_argument(
        "--sample_frac",
        type=float,
        default=None,
        help="Optional sub-sample fraction of the loaded dataset.",
    )
    p.add_argument("--seed", type=int, default=42)

    # Tokenization / truncation
    p.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory where the pre-tokenized dataset will be saved (save_to_disk).",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=65536,
        help="Truncate tokenized sequences to this length (default: 65536 to match run_sft.sh).",
    )
    p.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="Number of workers to use for dataset.map in this pre-tokenization step.",
    )
    p.add_argument(
        "--assistant_only_loss",
        action="store_true",
        help="If set, also compute assistant_masks matching TRL's assistant_only_loss behavior.",
    )

    # Sharding
    p.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="Number of shards to split the dataset into.",
    )
    p.add_argument(
        "--shard_index",
        type=int,
        default=None,
        help="Index of the shard to process.",
    )

    return p.parse_args()


def _tokenize_messages_example(
    example: dict[str, Any],
    tokenizer,
    assistant_only_loss: bool,
) -> dict[str, Any]:
    """Tokenize a single example containing a ChatML-style `messages` field."""
    messages = example["messages"]

    tools = example.get("tools")
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            tools = None

    processed = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=assistant_only_loss,
    )

    input_ids = processed["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]

    out: dict[str, Any] = {"input_ids": input_ids}

    if "assistant_masks" in processed:
        masks = processed["assistant_masks"]
        if masks and isinstance(masks[0], list):
            masks = masks[0]
        out["assistant_masks"] = masks

    return out


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    dataset: Dataset = load_converted_corpus(
        data_dir=args.data_dir,
        sources=args.sources,
        sample_frac=args.sample_frac,
        seed=args.seed,
    )

    if args.num_shards is not None and args.shard_index is not None:
        dataset = dataset.shard(num_shards=args.num_shards, index=args.shard_index)

    # Normalize conversations/messages structure to ChatML-style messages.
    dataset = dataset.map(
        maybe_convert_to_chatml,
        desc="Converting conversations to ChatML messages",
        num_proc=args.num_proc,
    )

    # Tokenize conversational messages to input_ids (and optional assistant_masks).
    original_columns = list(dataset.column_names)

    def tokenize_fn(example: dict[str, Any]) -> dict[str, Any]:
        return _tokenize_messages_example(
            example,
            tokenizer=tokenizer,
            assistant_only_loss=args.assistant_only_loss,
        )

    cols_to_remove = [c for c in original_columns if c not in ("messages", "tools")]

    dataset = dataset.map(
        tokenize_fn,
        desc="Tokenizing dataset",
        num_proc=args.num_proc,
        remove_columns=cols_to_remove,
    )

    # After tokenization, keep only token-level fields that the trainer expects.
    keep_cols = {"input_ids", "assistant_masks"}
    drop_cols = [c for c in dataset.column_names if c not in keep_cols]
    if drop_cols:
        dataset = dataset.remove_columns(drop_cols)

    if args.max_length is not None:
        dataset = truncate_dataset(
            dataset,
            args.max_length,
            map_kwargs={"num_proc": args.num_proc},
        )

    dataset.save_to_disk(args.output_path)


if __name__ == "__main__":
    main()

