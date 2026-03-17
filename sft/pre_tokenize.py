import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

from data import load_converted_corpus

from trl.data_utils import maybe_convert_to_chatml, truncate_dataset

_DIAGNOSTICS_FILENAME = "tokenization_diagnostics.jsonl"


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

    # Diagnostics
    p.add_argument(
        "--num_diagnostic_samples",
        type=int,
        default=10,
        help="Number of decoded samples to save for inspection (0 to disable).",
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


def _adapt_messages(messages: list[dict], tokenizer) -> list[dict]:
    """Ensure reasoning_content reaches the tokenized output.

    If the chat template handles ``reasoning_content`` natively (e.g. Qwen3.5),
    return messages unchanged.  Otherwise merge it into ``content`` wrapped in
    ``<think>...</think>`` tags so models whose vocab includes those tokens
    (e.g. Qwen3-4B-Instruct-2507) still learn to reason.
    """
    template = getattr(tokenizer, "chat_template", "") or ""
    if "reasoning_content" in template:
        return messages
    adapted = []
    for msg in messages:
        if msg.get("reasoning_content") and msg.get("role") == "assistant":
            msg = dict(msg)
            reasoning = msg.pop("reasoning_content")
            content = msg.get("content", "") or ""
            thinking = f"<think>\n{reasoning}\n</think>"
            msg["content"] = f"{thinking}\n\n{content}".strip()
        adapted.append(msg)
    return adapted


def _build_assistant_masks(input_ids: list[int], tokenizer) -> list[int]:
    """Build assistant token masks by scanning for ChatML role boundaries.

    Qwen chat templates don't support HuggingFace's ``{% generation %}`` tag,
    so ``apply_chat_template(return_assistant_tokens_mask=True)`` returns
    all-zero masks.  Instead, we scan the tokenized output for
    ``<|im_start|>assistant\\n`` ... ``<|im_end|>`` boundaries and mark the
    content tokens plus the closing ``<|im_end|>`` as assistant (1), everything
    else as non-assistant (0).  Including ``<|im_end|>`` is critical so the model
    learns to produce the stop token.
    """
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_token_id = tokenizer.encode("assistant", add_special_tokens=False)[0]

    mask = [0] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if (
            input_ids[i] == im_start_id
            and i + 1 < len(input_ids)
            and input_ids[i + 1] == assistant_token_id
        ):
            # Skip <|im_start|> + "assistant" + "\n"
            content_start = i + 3
            j = content_start
            while j < len(input_ids) and input_ids[j] != im_end_id:
                j += 1
            for k in range(content_start, j + 1): # Include the closing <|im_end|>
                mask[k] = 1
            i = j + 1
        else:
            i += 1
    return mask


def _tokenize_messages_example(
    example: dict[str, Any],
    tokenizer,
    assistant_only_loss: bool,
) -> dict[str, Any]:
    """Tokenize a single example containing a ChatML-style `messages` field."""
    messages = _adapt_messages(example["messages"], tokenizer)

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
    )

    input_ids = processed["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]

    out: dict[str, Any] = {"input_ids": input_ids}

    if assistant_only_loss:
        out["assistant_masks"] = _build_assistant_masks(input_ids, tokenizer)

    return out


def save_diagnostics(
    dataset: Dataset,
    tokenizer,
    output_dir: str | Path,
    num_samples: int = 10,
    seed: int = 42,
) -> Path:
    """Decode random samples from a tokenized dataset and save as JSONL.

    Each line contains the sample index, token count, and the full decoded
    text so you can inspect exactly what the model sees during training.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / _DIAGNOSTICS_FILENAME

    n = min(num_samples, len(dataset))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(dataset)), n))

    with open(out_path, "w") as f:
        for idx in indices:
            ids = dataset[idx]["input_ids"]
            decoded = tokenizer.decode(ids)
            record = {
                "index": idx,
                "num_tokens": len(ids),
                "decoded_text": decoded,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {n} decoded diagnostic samples to {out_path}")
    return out_path


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

    if args.num_diagnostic_samples > 0:
        save_diagnostics(
            dataset,
            tokenizer,
            output_dir=args.output_path,
            num_samples=args.num_diagnostic_samples,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

