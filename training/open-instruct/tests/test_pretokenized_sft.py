"""Pretokenized trajectories retain tool tokens and labels without retokenizing."""

import ast
import json
from pathlib import Path

import datasets


def loader():
    source = Path(__file__).parents[1] / "open_instruct/finetune.py"
    definition = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "load_pretokenized_jsonl"
    )
    namespace = {"datasets": datasets, "json": json}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[definition.name]


def test_tokens_labels_and_attention_are_preserved(tmp_path):
    path = tmp_path / "trajectories.jsonl"
    row = {"input_ids": [101, 7, 202], "labels": [-100, 7, 202]}
    path.write_text(json.dumps(row) + "\n\n")
    dataset = loader()(str(path), 3)
    assert len(dataset) == 1
    assert dataset[0] == {**row, "attention_mask": [1, 1, 1]}


def test_mismatched_and_overlong_records_are_filtered(tmp_path):
    rows = [
        {"input_ids": [1, 2], "labels": [1]},
        {"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4]},
        {"input_ids": [5, 6], "labels": [-100, 6]},
    ]
    path = tmp_path / "trajectories.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    dataset = loader()(str(path), 3)
    assert len(dataset) == 1
    assert dataset[0]["input_ids"] == [5, 6]
