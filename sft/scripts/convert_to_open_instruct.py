"""Convert osieosie/tmax-sft-full-20260403 → hamishivi/tmax-sft-full-20260403.

Merges reasoning_content and content into a single content field:
    <think>\n{reasoning_content}\n</think>\n\n{content}

Keeps role, tool_calls, tool_call_ids unchanged.
"""

import argparse
from datasets import load_dataset, DatasetDict


def convert_message(msg: dict) -> dict:
    """Merge reasoning_content into content."""
    reasoning = msg.get("reasoning_content") or ""
    content = msg.get("content") or ""

    if reasoning:
        if content:
            merged = f"<think>\n{reasoning}\n</think>\n\n{content}"
        else:
            merged = f"<think>\n{reasoning}\n</think>"
    else:
        merged = content

    out = {"role": msg["role"], "content": merged}

    # Preserve tool_calls and tool_call_ids if present
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    if msg.get("tool_call_ids"):
        out["tool_call_ids"] = msg["tool_call_ids"]

    return out


def convert_row(row: dict) -> dict:
    row["messages"] = [convert_message(m) for m in row["messages"]]
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="osieosie/tmax-sft-full-20260403")
    parser.add_argument("--dst", default="hamishivi/tmax-sft-full-20260403")
    parser.add_argument("--dry-run", action="store_true", help="Print samples instead of pushing")
    args = parser.parse_args()

    print(f"Loading {args.src}...")
    ds_dict = load_dataset(args.src)

    converted = DatasetDict()
    for split_name, ds in ds_dict.items():
        print(f"Converting {split_name} ({len(ds)} rows)...")
        converted[split_name] = ds.map(convert_row)

    if args.dry_run:
        # Print a sample from the first split
        split = list(converted.keys())[0]
        row = converted[split][0]
        for i, m in enumerate(row["messages"][:4]):
            print(f"\nmsg {i}: role={m['role']}")
            print(f"  content: {m['content'][:300]}")
            if m.get("tool_calls"):
                print(f"  tool_calls: {str(m['tool_calls'])[:200]}")
            if m.get("tool_call_ids"):
                print(f"  tool_call_ids: {m['tool_call_ids']}")
    else:
        print(f"Pushing to {args.dst}...")
        converted.push_to_hub(args.dst)
        print("Done!")


if __name__ == "__main__":
    main()
