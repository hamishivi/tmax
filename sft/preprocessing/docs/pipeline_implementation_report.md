# Technical Report: Terminus-2 to SWE-Agent Conversion Pipeline -- Implementation

**Date:** 2026-03-11
**Scope:** Complete implementation of the conversion pipeline specified in the [conversion specification report](terminus2_to_sweagent_conversion_report.md) (v2.1), including all design decisions, architectural choices, iterative bug fixes, and the final production-ready state. This report covers the full journey from specification to working pipeline.

**Prerequisite reading:** The [conversion specification](terminus2_to_sweagent_conversion_report.md) defines the source and target formats, conversion logic, and quality filtering criteria. This report documents how that specification was realized in code, what decisions were made along the way, and what issues were discovered and resolved during real-data validation.

---

## 1. Executive Summary

We implemented a concurrent preprocessing pipeline that converts ~381,000 Terminus-2 format agentic traces from two HuggingFace datasets into SWE-agent style tool-calling format, suitable for SFT training of Qwen3.5-family models.

The pipeline lives entirely within `sft/preprocessing/` and is cleanly separated from the downstream training code. Key metrics from the full production run:

| Metric | Value |
|--------|-------|
| Total input traces | ~381,000 across 5 sources |
| Final yield | ~85-90% after all filters |
| JSON extraction success | >99% (5-strategy cascade) |
| Pipeline runtime | ~25 min on 15-core node |
| Output format | Per-source Parquet files |

The pipeline went through five phases of iterative improvement after the initial implementation, each driven by data quality issues discovered during validation runs. All improvements are documented in the [pipeline improvements report](pipeline_improvements_report.md).

---

## 2. Architecture Overview

### 2.1 Separation of Concerns

The conversion pipeline is a **standalone preprocessing step** that runs before tokenization and training. This separation was a deliberate design choice:

- **Preprocessing** (`sft/preprocessing/`) downloads raw Terminus-2 data from HuggingFace, converts it to SWE-agent format, applies quality filters, and writes Parquet files.
- **Data loading** (`sft/data.py`) reads the converted Parquet output and prepares it for the training pipeline.
- **Tokenization** (`sft/pre_tokenize.py`) applies the Qwen3.5 chat template to produce token IDs.
- **Training** (`sft/train.py`) consumes pre-tokenized data via TRL's `SFTTrainer`.

This separation means the conversion only needs to run once (or when the pipeline logic changes), and the converted data can be re-tokenized with different models or templates without re-running conversion.

### 2.2 Module Structure

```
sft/preprocessing/
    __init__.py              # Lazy-loading entry point (avoids pyarrow dep for tests)
    json_extraction.py       # 5-strategy JSON extraction cascade
    builders.py              # Pure builder functions for messages, tool_calls, tool results
    convert.py               # Core convert_trace() function
    filters.py               # Mandatory, warning, and optional quality filters
    pipeline.py              # CLI orchestrator, I/O, statistics, reporting
    report.py                # Rich terminal + plain-text report generation
    config/
        system_prompt.txt    # Replacement system prompt (loaded at runtime)
        tool_schemas.json    # bash tool definition (injected into training data)
        sources.yaml         # Declarative source registry
    docs/
        terminus2_to_sweagent_conversion_report.md   # v2.1 spec
        pipeline_improvements_report.md               # Iterative fixes
        pipeline_implementation_report.md             # This document
```

### 2.3 Data Flow

```
HuggingFace Hub
    |
    v
pipeline.py (download + concurrent pre-fetch)
    |
    v
convert.py (per-trace: parse msg-0, walk turns, build messages)
    |   calls: json_extraction.py, builders.py
    v
filters.py (mandatory drops, optional drops, warning flags)
    |
    +---> kept traces -----> per-source Parquet files
    |                            |
    +---> dropped traces ---> *_dropped.jsonl
    |                            |
    +---> conversion_report.json + conversion_report.txt
                                     |
                                     v
                            data.py (load Parquet, inject tools column)
                                     |
                                     v
                            pre_tokenize.py (apply_chat_template -> token IDs)
                                     |
                                     v
                            train.py (SFTTrainer)
```

---

## 3. Resolution of Specification Open Questions

The v2.1 specification (Section 11) listed 10 open questions. Here is how each was resolved during implementation:

| # | Question | Resolution |
|---|----------|------------|
| 1 | **System prompt wording** | Finalized to match the inference harness prompt. Describes persistent bash terminal, lists rules (one command per turn, `timeout` for long commands, `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` for submission). Stored in `config/system_prompt.txt`. |
| 2 | **`thought` vs `reasoning_content`** | `reasoning_content` is used directly. The Qwen3.5 chat template reads this field and renders it as `<think>...</think>` blocks. |
| 3 | **Tool result `content` format** | Flat string (`"content": "terminal output..."`) rather than list-of-content-blocks. Qwen3.5's template expects a string for tool results. |
| 4 | **Mixing with native SWE-agent data** | Not addressed at pipeline level -- deferred to training-time data mixing configuration. |
| 5 | **Submit tool result** | Eliminated the `submit` tool entirely. Task completion is signaled via `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` as a bash command. This matches the inference harness protocol and avoids the need for a synthetic tool result. |
| 6 | **`duration` field** | Discarded. Not meaningful for the target format. |
| 7 | **Special keystroke protocol** | `C-c` and `C-d` are preserved as literal strings in the command field. Traces containing `C-c` are filtered out during quality filtering (we do not want the SFT model to learn interrupt behavior). |
| 8 | **Data mixing ratio** | No rebalancing at pipeline level. The ~24:1 Nemotron-to-OpenThoughts ratio is preserved. Rebalancing is deferred to training configuration (sampling weights). |
| 9 | **`dataset_adapters` complexity** | These traces convert cleanly. The main quality issue was the `<think>` tag wrapping (see Section 5), which affected all Nemotron subsets equally. |
| 10 | **License compatibility** | Not addressed at pipeline level. Nemotron uses CC-BY-4.0; OpenThoughts license is verified separately. |

---

## 4. Key Design Decisions Made During Implementation

### 4.1 YAML Source Registry

All source datasets are defined declaratively in `config/sources.yaml`. Each entry specifies the HuggingFace repo name, loading method (`huggingface` for standard datasets, `huggingface_parquet` for repos with large row groups), glob patterns for parquet files, and a `source_label` string.

Adding a new Terminus-2 dataset requires only editing this YAML file -- no code changes. The `source_label` is stamped on every converted trace as the `source` column, enabling downstream filtering by provenance.

### 4.2 Single `source` Column

Every converted trace carries a `source` column with a full provenance path (e.g., `"nvidia/Nemotron-Terminal-Corpus/skill_based_easy"`). This was chosen over separate `source_dataset` + `source_subset` columns for simplicity. The `data.py` loader supports filtering by source label at load time.

### 4.3 Submission via Bash Echo

The specification originally proposed a separate `submit` tool. During implementation, this was changed to use the inference harness's actual protocol: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` as a bash command. This means:

- Only one tool (`bash`) is defined in the tool schema.
- When `task_complete: true` is encountered, the pipeline emits a bash tool_call with the echo command.
- The conversion loop `break`s immediately after the first submit, discarding any confirmation-loop messages.

### 4.4 Terminal State Discarded from First User Message

The specification (Section 5.2) noted that the initial terminal state (40 lines of mostly-blank terminal buffer) should be preserved. During data quality review, we determined this was container-specific noise (UUID hostnames, empty buffer lines). The terminal state section is now stripped from the first user message -- only the task description is kept.

### 4.5 No Turn Limit

The specification (Section 7.3) suggested dropping traces with >20 turns. After analyzing the data, we found that long traces in the Nemotron corpus represent legitimate complex tasks. The `max_turns` filter was set to 999 (effectively disabled), with the CLI flag available if needed.

### 4.6 64K Token Maximum

The training pipeline uses a 65,536 token maximum sequence length, matching the Qwen3.5 context window. Pre-tokenization truncates at this boundary.

---

## 5. JSON Extraction: The 5-Strategy Cascade

The most complex component is `json_extraction.py`, which must reliably extract structured JSON from assistant messages that may contain prose, `<think>` tags, and malformed JSON. The cascade evolved from 3 strategies (specification Section 5.4) to 5 strategies after encountering real Nemotron data.

### Strategy 1: Direct Parse
Content is pure JSON. `json.loads()` succeeds directly. This handles ~80% of OpenThoughts traces.

### Strategy 2: Brace-Match + `raw_decode`
JSON is embedded in prose. Uses Python's `json.JSONDecoder.raw_decode()` starting from the first `{` to find and parse the JSON object. More efficient than manual brace-matching for well-formed JSON.

### Strategy 3: Error Fix + Retry
When `raw_decode` fails (e.g., trailing commas from LLM generation errors), falls back to manual brace-matching with `_find_matching_brace()`, applies `_fix_common_json_errors()` (removes trailing commas before `}` or `]`), and retries.

### Strategy 4: `<think>` Tag Stripping (Post-Tag JSON)
The Nemotron teacher model (DeepSeek-V3.2) wraps all assistant content in `<think>...</think>` tags. Strategy 4 strips the tags and applies strategies 1-3 to the text after `</think>`. This handles the majority of Nemotron traces where the JSON lives after the closing tag.

### Strategy 5: Merged Prose+JSON Inside `<think>` (Regex Extraction)
A subset of Nemotron traces have no clean JSON boundary -- the model's reasoning prose flows directly into the JSON value body. The `analysis` field's opening `{` and key are missing; only `plan`, `commands`, and `task_complete` remain as parseable fields. Strategy 5 uses targeted regex parsing to extract these fields individually:
- `commands`: bracket-matching for the `[...]` array
- `task_complete`: regex for `true`/`false` value
- `plan`: string extraction with escape-aware quote matching
- `analysis`: everything before the first extracted key

### Yield Improvement

| Phase | Yield | Primary Recovery |
|-------|-------|-----------------|
| Strategies 1-3 only | 30.6% on 1% sample | -- |
| + Strategy 4 | ~65% | `<think>`-wrapped JSON after `</think>` |
| + Strategy 5 | ~70% | Merged prose+JSON inside `<think>` |
| + Other fixes (Sections 6-7) | ~85-90% | Harness errors, C-c, incomplete traces |

---

## 6. Builder Functions

### 6.1 `build_reasoning_content`
Combines the `analysis` field, `plan` field, and any surrounding prose (text outside the JSON blob) into a single string. This becomes the `reasoning_content` field on assistant messages, rendered as `<think>...</think>` by the Qwen3.5 template.

A critical fix here was ensuring that `<think>` tags from the source data are stripped before populating `reasoning_content`. Without this, the template would produce double-wrapped `<think><think>content</think></think>` during tokenization. After the fix, zero `<think>` tag leaks were verified across the full dataset.

### 6.2 `build_tool_calls`
Converts the `commands` array into a single `bash` tool_call. Multiple keystrokes are newline-joined. Trailing `\n` (the "Enter" key in Terminus-2) is stripped. Special keystrokes (`C-c`, `C-d`) are preserved verbatim. Wait-only commands (empty keystrokes with `duration`) are dropped. Tool call IDs are deterministic SHA-256 hashes of `conversation_id:turn_index`.

### 6.3 `build_tool_result`
Converts terminal output into a flat-string tool result. Strips:
- Harness framing prefixes (`"Current terminal state:\n"`, `"New Terminal Output:\n"`)
- The "Are you sure you want to mark the task as complete?" confirmation
- Container-specific shell prompts (`root@<uuid-hostname>:/path#`) via a regex that matches hostnames >= 7 characters
- Trailing blank lines from the 40-line terminal buffer

The shell prompt stripping reduces tool output size by ~24% on typical messages while preserving actual command output, heredoc content, and continuation prompts.

### 6.4 `build_submit_messages`
Emits a `bash` tool_call with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` when `task_complete: true`. If the turn had no other commands, the reasoning is attached to the submit message.

### 6.5 `is_harness_error`
Detects Nemotron harness error messages injected when the agent's JSON response could not be parsed. These messages contain markers like `"ERROR: No valid JSON found"`. When detected, the entire turn (assistant + error response) is skipped during conversion.

---

## 7. Conversion Loop (`convert_trace`)

The core function processes a single trace through these steps:

1. **Parse Message 0**: Split on `"Task Description:\n"` to separate system prompt from task. Split on `"Current terminal state:\n"` to extract (and discard) terminal state.

2. **Emit system + user**: Replace the Terminus-2 system prompt with the configured system prompt. Emit the task description as the first user message.

3. **Walk (assistant, user) pairs**:
   - Extract JSON via the 5-strategy cascade
   - Build `reasoning_content` from analysis + plan + prose
   - Build `tool_calls` from commands
   - **Harness error check**: If the next user message is a harness error, skip the entire turn
   - **Reasoning buffering**: If an assistant turn has reasoning but no commands, buffer it instead of emitting a standalone message. Prepend the buffered reasoning to the next assistant turn that has commands. This collapses non-standard consecutive-assistant sequences.
   - **C-c tracking**: Flag traces containing Ctrl+C for downstream filtering
   - **Submit truncation**: When `task_complete: true`, emit the submit bash command and `break` -- all subsequent messages (confirmation prompts, re-confirmation) are discarded

4. **Build metadata**: Source model, task ID, trial name, turn count, strategy distribution, quality flags.

---

## 8. Quality Filtering

### Mandatory Filters (trace dropped)

| Filter | Rationale |
|--------|-----------|
| `conversion_failed` | Trace failed basic validation (e.g., doesn't start with user message) |
| `json_extraction_failed` | At least one assistant turn had no extractable JSON |
| `too_few_turns` | Fewer than 1 agent turn -- no behavior to learn from |
| `no_task_complete` | Agent never marked task as complete (configurable via `--include-partial`) |
| `contains_ctrl_c` | Agent used Ctrl+C to interrupt -- undesirable SFT signal |

### Warning Flags (kept but flagged)

| Flag | Rationale |
|------|-----------|
| `no_task_complete` | When `--include-partial` is used, incomplete traces are kept with this warning |
| `task_delim_missing` | System prompt / task boundary not found in message 0 |
| `prose_outside_json` | Assistant turn had text outside the JSON blob |
| `missing_tool_result` | Assistant had commands but no following user message |

### Optional Filters (configurable)

| Filter | Default |
|--------|---------|
| `max_turns` | 999 (effectively no limit) |
| `drop_trivial_only` | False (keep trivial-command-only traces) |

---

## 9. Pipeline Orchestrator

### 9.1 Concurrent Pre-Fetch
Downloads for all 5 source datasets happen concurrently via `ThreadPoolExecutor` (I/O-bound HF Hub downloads overlap). This eliminates sequential download latency.

### 9.2 Single-Pass Processing
Initial implementation used `datasets.map(num_proc=N)` for parallel conversion. Profiling revealed that `convert_trace` itself takes <1 ms per row -- the bottleneck was Arrow serialization/deserialization overhead (99.6% of wall time). Multiprocessing made it *slower* due to fork + IPC cost.

The production implementation uses a single-pass pure-Python loop: bulk-decode all Arrow columns to Python once, iterate in pure Python (zero Arrow overhead per row), and construct output Datasets from plain lists at the end. This is approximately 4x faster than the `datasets.map` approach.

### 9.3 Sharding Support
For very large datasets, the pipeline supports distributed processing via `--shard-index` and `--num-shards` flags. Each shard processes a contiguous slice of each source. A `merge_shards` function concatenates per-shard Parquet files and aggregates statistics.

### 9.4 Reporting
After processing, the pipeline generates:
- `conversion_report.json`: Machine-readable statistics with per-source breakdowns
- `conversion_report.txt`: Human-readable ANSI-stripped report
- Rich terminal output with colored tables showing per-source yield, drop-reason breakdowns with bar charts, JSON strategy distribution, turn-count distribution, warning flag counts, and qualitative example traces

---

## 10. Integration with Training Pipeline

### 10.1 `data.py` Refactoring
The original `data.py` downloaded raw Nemotron data directly from HuggingFace. It was refactored to:
- Load converted Parquet files from the preprocessing output directory
- Filter by `source` column (e.g., load only specific subsets)
- Inject a constant `tools` column (the bash tool schema as a JSON string) so that `pre_tokenize.py` can pass it to `apply_chat_template(tools=...)`
- Support sub-sampling via `sample_frac`

### 10.2 `pre_tokenize.py` Updates
- Uses `load_converted_corpus()` instead of the old `load_terminal_corpus()`
- Accepts `--data_dir` and `--sources` arguments
- The `tools` column from `data.py` feeds into `apply_chat_template(tools=...)` for correct Qwen3.5 tool-call rendering

### 10.3 `train.py` Updates
- Fallback data loading path uses `load_converted_corpus()`
- Default `max_length` set to 65,536 (64K tokens)
- Primary path (pre-tokenized `load_from_disk`) unchanged

---

## 11. Testing

### Unit Tests (56 passing)

| Test File | Coverage |
|-----------|----------|
| `test_json_extraction.py` | Strategies 1-3: clean JSON, prose+JSON, trailing commas, nested braces, empty string, non-dict JSON |
| `test_builders.py` | `build_reasoning_content`: both fields, analysis-only, with prose, empty. `build_tool_calls`: single/multiple commands, C-c/C-d, mixed, wait-only, empty, deterministic IDs, dict arguments, quoted commands. `build_tool_result`: standard output, confirmation stripped, flat string, cwd prompts preserved. `build_submit_messages`: with/without commands, task_complete true/false/absent, deterministic IDs. |
| `test_convert.py` | Full trace round-trip, system prompt replacement, task description preservation, `reasoning_content` present, tool_call structure, tool result structure, submit present, metadata, source label. Edge cases: empty conversations, non-user start, missing delimiters, malformed JSON, special keystrokes, confirmation turns. Structural integrity: every tool_call has matching result, no duplicate IDs, correct role ordering. |

### Real-Data Validation

All fixes were validated on the 1% teaser sample (3,812 traces):
- 5,220 of 5,253 previously-failing messages recovered (99.3%)
- 100% of recovered messages pass through downstream builders without error
- Zero `<think>` tag leaks in `reasoning_content`
- Full production run (381K traces) completed without errors

---

## 12. Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_conversion.sh` | Full pipeline on all sources. Supports `--upload` to push to HuggingFace, `--include-partial` for truncated traces. |
| `scripts/run_conversion_teaser.sh` | 1% sample for quick iteration. Separate output directory (`terminus2_sweagent_1pct`). |
| `scripts/run_pretokenize_sweagent_full_qwen3.sh` | Pre-tokenize the full converted dataset with Qwen3.5-4B tokenizer. |
| `scripts/upload_data_to_hf.sh` | Standalone upload of converted data to HuggingFace Hub. |

---

## 13. File Manifest

### New Files Created

| File | Purpose |
|------|---------|
| `sft/preprocessing/__init__.py` | Package init with lazy pipeline import |
| `sft/preprocessing/json_extraction.py` | 5-strategy JSON extraction cascade |
| `sft/preprocessing/builders.py` | Builder functions for messages, tool_calls, results, submit |
| `sft/preprocessing/convert.py` | Core `convert_trace()` with reasoning buffering, harness error deletion, submit truncation |
| `sft/preprocessing/filters.py` | Mandatory, warning, and optional quality filters |
| `sft/preprocessing/pipeline.py` | CLI orchestrator with concurrent pre-fetch, single-pass processing, sharding, reporting |
| `sft/preprocessing/report.py` | Rich terminal + plain-text report generation |
| `sft/preprocessing/config/system_prompt.txt` | Replacement system prompt |
| `sft/preprocessing/config/tool_schemas.json` | Bash tool schema for Qwen3.5 chat template |
| `sft/preprocessing/config/sources.yaml` | YAML source registry (2 datasets, 5 subsets) |
| `sft/tests/test_json_extraction.py` | Unit tests for JSON extraction |
| `sft/tests/test_builders.py` | Unit tests for builder functions |
| `sft/tests/test_convert.py` | Integration tests for full trace conversion |
| `sft/scripts/run_conversion.sh` | Full pipeline launcher |
| `sft/scripts/run_conversion_teaser.sh` | 1% teaser launcher |

### Modified Files

| File | Changes |
|------|---------|
| `sft/data.py` | Refactored: raw HF downloading removed, replaced with `load_converted_corpus()` that reads converted Parquet, injects tools column, supports source filtering |
| `sft/pre_tokenize.py` | Updated to use `load_converted_corpus()`, added `--data_dir` and `--sources` args |
| `sft/train.py` | Updated fallback loading to `load_converted_corpus()`, default `max_length` set to 65536 |
| `.gitignore` | Added `sft/preprocessing/terminus2_sweagent/` and `sft/preprocessing/terminus2_sweagent_1pct/` |

---

## 14. Known Limitations and Future Work

1. **Tool output command echo.** Terminal output includes the typed command echoed back (redundant with the `tool_call` command). Stripping echoes from interleaved multi-command output is fragile and was intentionally deferred.

2. **Data mixing ratio.** The ~24:1 Nemotron-to-OpenThoughts ratio is not rebalanced. Training-time sampling weights or curriculum learning should handle this.

3. **SWE-Bench patch format tasks.** Some `dataset_adapters` traces involve SWE-bench-style patch generation. These are kept in the dataset but may have different quality characteristics.

4. **Chat template verification.** The specification (Section 6) recommends running every converted trace through the Qwen3.5 chat template before training. This is handled at the `pre_tokenize.py` stage; traces that fail template rendering are caught there, not at conversion time.

5. **No `submit` tool result.** When the model echoes the submit command, there is no following tool result message. This matches the inference harness behavior (the harness terminates the session on submit) but means the last message in every trace is an assistant message with a tool_call that has no response.
