#!/usr/bin/env python3
"""Chart the rubric-label distribution for the tmax-15k triage pass.

Form: the data's job is magnitude-across-categories, so horizontal bars sorted
by count, plus a hero number for the headline (keep rate). Deliberately NOT a
pie: comparing angles is the classic anti-pattern for this job.

Colour: a single series gets a single hue, so identity is never carried by
colour alone. The keep categories are marked with an explicit text badge rather
than a second hue. Values use text tokens, not the series colour. Hex values are
the documented, pre-validated categorical slots (slot 1 / slot 2) from the
dataviz reference palette -- `node` is unavailable on this cluster so the
validator could not be re-run, and inventing colours would be the one thing the
skill tells you never to do.

  python3 plot_labels.py --labels labels.jsonl --out labels.png
"""

from __future__ import annotations

import argparse
import collections
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SERIES_1 = "#2a78d6"  # slot 1, documented-validated
SERIES_2 = "#eb6834"  # slot 2, documented-validated (adjacent pair to slot 1)
KEEP_LABELS = {"CLEAN", "TASK-TRIVIAL"}


def load(path: str) -> list[dict]:
    rows = []
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("label"):
            rows.append(r)
    return rows


def hbar(ax, pairs, title, subtitle, highlight: set[str]) -> None:
    labels = [k for k, _ in pairs][::-1]
    vals = [v for _, v in pairs][::-1]
    total = sum(vals) or 1
    colors = [SERIES_2 if l in highlight else SERIES_1 for l in labels]
    bars = ax.barh(range(len(labels)), vals, color=colors, height=0.62)
    # 4px-equivalent rounded data-ends are not available in matplotlib bars;
    # keep marks thin and let the value labels carry precision instead.
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(
        [f"{l}  ·KEEP" if l in highlight else l for l in labels],
        fontsize=9, color=TEXT_PRIMARY,
    )
    ax.set_xlabel("tasks", fontsize=8, color=TEXT_SECONDARY)
    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left", pad=24, weight="bold")
    ax.text(0, 1.055, subtitle, transform=ax.transAxes, fontsize=8,
            color=TEXT_SECONDARY, va="bottom")
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + total * 0.012, b.get_y() + b.get_height() / 2,
                f"{v}  ({100*v/total:.1f}%)", va="center", fontsize=8,
                color=TEXT_SECONDARY)
    ax.set_xlim(0, max(vals) * 1.28)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#dcdcd8")
    ax.tick_params(axis="x", colors=TEXT_SECONDARY, labelsize=8, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#ececE8", linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-5-6-sol-genai-responses")
    args = ap.parse_args()

    rows = load(args.labels)
    n = len(rows)
    labs = collections.Counter(r["label"] for r in rows)
    hows = collections.Counter(
        (r.get("verifier_validates_how") or "unstated").split(";")[0].strip()[:22]
        for r in rows
    )
    keep = sum(v for k, v in labs.items() if k in KEEP_LABELS)

    fig = plt.figure(figsize=(12.4, 7.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 4.2], hspace=0.42, wspace=0.34,
                          left=0.20, right=0.97, top=0.90, bottom=0.08)

    # Hero number: the headline is a single value, so it is not a chart.
    hero = fig.add_subplot(gs[0, :])
    hero.axis("off")
    hero.text(0, 0.58, f"{100*keep/max(n,1):.1f}%", fontsize=38, weight="bold",
              color=SERIES_2, va="center")
    hero.text(0.20, 0.78, "of graded tasks are keepers (CLEAN + TASK-TRIVIAL)",
              fontsize=12, color=TEXT_PRIMARY, va="center")
    hero.text(0.20, 0.42,
              f"{keep} of {n} graded  ·  grader {args.model}  ·  one pass, no ensemble",
              fontsize=9, color=TEXT_SECONDARY, va="center")

    ax1 = fig.add_subplot(gs[1, 0], facecolor=SURFACE)
    hbar(ax1, labs.most_common(), "Rubric label",
         "why a task is or is not usable as RL training data", KEEP_LABELS)

    ax2 = fig.add_subplot(gs[1, 1], facecolor=SURFACE)
    hbar(ax2, hows.most_common(10), "How the verifier validates",
         "self-reported by the grader; explains the label distribution", set())

    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    print(f"wrote {args.out}  (n={n})")
    for k, v in labs.most_common():
        print(f"  {v:6d}  {100*v/n:5.1f}%  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
