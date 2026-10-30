# TMax v1.1 evaluation rollouts

Evaluation conversations for the five [released TMax v1.1 models](https://huggingface.co/collections/TMaxxx/tmax-v11), using their `main` checkpoints selected on TB-Lite. The 15 runs contain **4,095 accepted trials**, with three trials per task.

| Model | Training step | TB-Lite | Terminal-Bench 2.1 | Terminal-Bench 2.0 |
|---|---:|---:|---:|---:|
| [2B](https://huggingface.co/TMaxxx/tmax-2b-v1.1) | 400 | 17.78% | 5.30% | 6.37% |
| [4B](https://huggingface.co/TMaxxx/tmax-4b-v1.1) | 500 | 54.27% | 24.62% | 25.09% |
| [8B](https://huggingface.co/TMaxxx/tmax-8b-v1.1) | 500 | 17.78% | 6.44% | 8.61% |
| [9B](https://huggingface.co/TMaxxx/tmax-9b-v1.1) | 200 | 65.62% | 29.55% | 32.96% |
| [27B](https://huggingface.co/TMaxxx/tmax-27b-v1.1) | 400 | 76.93% | 46.21% | 43.82% |

The teaser's 2B and 27B Terminal-Bench 2.0 scores come from separate evaluations that preserve thinking; the 27B teaser evaluation uses step 200. This table and the conversations here cover the released main checkpoints.

These runs use Vanillux 0.2.3: Sandfleet for TB-Lite and Terminal-Bench 2.1, and Daytona for Terminal-Bench 2.0. TB-Lite has 96 tasks, Terminal-Bench 2.1 has 88 tasks (`configure-git-webserver` excluded), and Terminal-Bench 2.0 has 89 tasks. Terminal-Bench 2.0 agent-timeout attempts were retired and rerun at concurrency four.

Each `tmax-{size}b/` directory contains `tb-lite/`, `terminal-bench-2.1/`, and `terminal-bench-2.0/`. A benchmark directory has a `summary.json` and one directory per accepted trial:

- `agent/trajectory.json`: conversation messages and tool calls.
- `agent/usage.json`: token usage, cost, and step limit recorded by the harness.
- `result.json`: task and trial IDs, raw and effective rewards, scoring overrides, exception type, and the published trajectory's SHA256.

Scores average the three effective rewards within each task, then equally across tasks. `raw_reward` preserves the original grader output; `effective_reward` includes reviewed model-caused failures or accepted grading recovery. `score_override` preserves the explanation. Each summary includes the score, standard error, exception counts, and redaction counts. Retried attempts and duplicate copies were resolved before export using the original accepted scoring trial list.

[`manifest.json`](manifest.json) records immutable evaluated checkpoint revisions and verifies that their model weight shards match the released step branches and `main`. Its weight fingerprints hash the sorted filename-to-LFS-SHA256 maps. [`SHA256SUMS`](SHA256SUMS) covers every published file in this directory, including the manifest.

Private execution paths, account names, runtime identifiers, host filesystem/capacity metadata, operational details in scoring explanations, and service credential values are redacted. Benchmark fixture paths, public library/provider names, and the synthetic token explicitly declared by the `sanitize-git-repo` benchmark tests are preserved. Execution configs, host metadata, operational logs, and original receipts are omitted. The conversations otherwise preserve the original message content and structure; checksums refer to the published files after redaction.
