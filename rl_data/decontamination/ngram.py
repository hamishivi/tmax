"""Tiny word-n-gram helpers for the decontamination eval.

Tokenisation is intentionally aggressive (lowercase + split on any
non-alphanumeric run) so that punctuation, markdown, and whitespace
variation don't mask near-identical task descriptions.
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator, Sequence, Set, Tuple

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")

NGram = Tuple[str, ...]


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_SPLIT.split(text.lower()) if t]


def iter_ngrams(tokens: Sequence[str], n: int, stride: int = 1) -> Iterator[NGram]:
    if n <= 0 or stride <= 0 or len(tokens) < n:
        return
    for i in range(0, len(tokens) - n + 1, stride):
        yield tuple(tokens[i : i + n])


def build_index(texts: Iterable[str], n: int, stride: int = 1) -> Set[NGram]:
    """Union of all *n*-grams across *texts* (post-tokenisation)."""
    index: Set[NGram] = set()
    for text in texts:
        for ng in iter_ngrams(tokenize(text), n, stride):
            index.add(ng)
    return index


def scan_doc(text: str, index: Set[NGram], n: int, stride: int = 1) -> tuple[int, int]:
    """Return (hits, total) n-gram counts for *text* against *index*."""
    hits = 0
    total = 0
    for ng in iter_ngrams(tokenize(text), n, stride):
        total += 1
        if ng in index:
            hits += 1
    return hits, total
