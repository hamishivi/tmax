"""Data loading for SFT training.

Loads pre-processed (converted) data from the preprocessing pipeline output.
The converted Parquet files contain ``messages`` (SWE-agent format),
``source`` (provenance label), and ``metadata`` columns.

This module injects the ``tools`` column (constant tool schemas for ``bash``
and ``submit``) so that ``pre_tokenize.py`` can pass them to
``apply_chat_template(tools=...)``.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset

from preprocessing.convert import get_tool_schemas

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "preprocessing" / "terminus2_sweagent"

TRAIN_COLUMNS = {"messages", "tools"}


def load_converted_corpus(
    data_dir: str | Path | None = None,
    sources: list[str] | None = None,
    sample_frac: float | None = None,
    seed: int = 42,
) -> Dataset:
    """Load converted SWE-agent format data for SFT training.

    Parameters
    ----------
    data_dir : path
        Directory containing the Parquet files produced by the conversion
        pipeline (default: ``sft/preprocessing/terminus2_sweagent``).
    sources : list[str] | None
        Source labels to include (e.g.
        ``["nvidia/Nemotron-Terminal-Corpus/skill_based_easy"]``).
        If ``None``, all Parquet files in *data_dir* are loaded.
    sample_frac : float | None
        If set, randomly sub-sample this fraction of the final dataset.
    seed : int
        Random seed for sub-sampling.

    Returns
    -------
    Dataset with ``messages`` and ``tools`` columns, ready for tokenization.
    """
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR
    data_dir = Path(data_dir)

    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No .parquet files found in {data_dir}.  "
            "Run the conversion pipeline first: python -m preprocessing.pipeline"
        )

    ds = load_dataset(
        "parquet",
        data_files=[str(p) for p in parquet_files],
        split="train",
    )

    # Filter by source label if requested
    if sources is not None:
        source_set = set(sources)
        ds = ds.filter(
            lambda row: row["source"] in source_set,
            desc="Filtering by source",
        )

    # Sub-sample
    if sample_frac is not None and 0 < sample_frac < 1:
        n = max(1, int(len(ds) * sample_frac))
        ds = ds.shuffle(seed=seed).select(range(n))

    # Inject constant tools column
    tool_schemas = get_tool_schemas()
    tool_schemas_str = json.dumps(tool_schemas)
    ds = ds.map(
        lambda row: {"tools": tool_schemas_str},
        desc="Injecting tool schemas",
    )

    # Keep only training-relevant columns
    drop = [c for c in ds.column_names if c not in TRAIN_COLUMNS]
    if drop:
        ds = ds.remove_columns(drop)

    return ds


# Backward-compat alias
load_terminal_corpus = load_converted_corpus
