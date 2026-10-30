# TMax v1.1 evaluation rollouts

Terminal-Bench 2.0 evaluation conversations from the Daytona runs for the five [released TMax v1.1 models](https://huggingface.co/collections/TMaxxx/tmax-v11), using their `main` checkpoints selected on TB-Lite. The five runs contain **1,335 accepted trials**: all 89 tasks, with three trials per task for each model.

| Model | Training step | Terminal-Bench 2.0 |
|---|---:|---:|
| [2B](https://huggingface.co/TMaxxx/tmax-2b-v1.1) | 400 | 6.37% |
| [4B](https://huggingface.co/TMaxxx/tmax-4b-v1.1) | 500 | 25.09% |
| [8B](https://huggingface.co/TMaxxx/tmax-8b-v1.1) | 500 | 8.61% |
| [9B](https://huggingface.co/TMaxxx/tmax-9b-v1.1) | 200 | 32.96% |
| [27B](https://huggingface.co/TMaxxx/tmax-27b-v1.1) | 400 | 43.82% |

The teaser's 2B and 27B Terminal-Bench 2.0 scores come from separate evaluations that preserve thinking; the 27B teaser evaluation uses step 200. This table and the conversations here cover the released main checkpoints.

These runs use Vanillux 0.2.3 with the Daytona backend. Agent-timeout attempts were retired and rerun at concurrency four; only accepted scoring trials are included.

## Task coverage

All **89 Terminal-Bench 2.0 tasks** are included for every model, including `configure-git-webserver`; no tasks were excluded. Each model has **267 trials**, and scores and standard errors use the full 89-task set. This archive contains only the Daytona Terminal-Bench 2.0 runs.

## Archive contents

The `daytona/` directory groups the runs by model, following the v1.0 archive layout. Each `daytona/tmax-{size}b/terminal-bench-2-0/` directory has a `summary.json` and one directory per accepted trial:

- `agent/trajectory.json`: conversation messages and tool calls.
- `agent/usage.json`: token usage, cost, and step limit recorded by the harness.
- `result.json`: task and trial IDs, raw and effective rewards, scoring overrides, exception type, and the published trajectory's SHA256.

Scores average the three effective rewards within each task, then equally across tasks. `raw_reward` preserves the original grader output; `effective_reward` includes reviewed model-caused failures or accepted grading recovery. `score_override` preserves the explanation. Each summary includes the score, standard error, exception counts, and redaction counts. Retried attempts and duplicate copies were resolved before export using the original accepted scoring trial list.

[`manifest.json`](manifest.json) records immutable evaluated checkpoint revisions and verifies that their model weight shards match the released step branches and `main`. Its weight fingerprints hash the sorted filename-to-LFS-SHA256 maps. [`SHA256SUMS`](SHA256SUMS) covers every published file in this directory, including the manifest.

Private execution paths, account names, runtime identifiers, host filesystem/capacity metadata, operational details in scoring explanations, and service credential values are redacted. Benchmark fixture paths, public library/provider names, and the synthetic token explicitly declared by the `sanitize-git-repo` benchmark tests are preserved. Execution configs, host metadata, operational logs, and original receipts are omitted. The conversations otherwise preserve the original message content and structure; checksums refer to the published files after redaction.
