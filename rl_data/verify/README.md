# Task audit, repair, and optional solvability filtering

This is a public restoration of the historical TMax audit tools, with a later
direct-provider client and explicitly reconstructed image-build orchestration.
It is **not an identical replay of the paper run**. See the
[provenance report](../../docs/task-audit-provenance.md) for the file-by-file
classification, released artifact checks, and missing historical evidence.

The paper's **13,565-task TMax-13K v1.1** is the hardened **prefilter**: applied
repairs plus original `CLEAN`/`TASK-TRIVIAL` tasks. The **7,015-task** subset with
a demonstrated passing solution is a separate artifact. Two failed attempts do
not establish unsolvability. Neither fresh provider calls nor these public code
changes reproduce historical verdicts automatically.

## Use the released tasks

For training, use [TMaxxx/tmax-13k-v1.1](https://huggingface.co/datasets/TMaxxx/tmax-13k-v1.1).
Its [metadata companion](https://huggingface.co/datasets/TMaxxx/tmax-13k-v1.1-metadata)
contains audit columns and `image-recipe/`. Use those published recipes or the
released Docker images to recover the historical repaired environments. Do not
substitute a newly resolved dependency plan and call it the historical image.

Independently check the pinned artifacts (downloads task data and metadata,
roughly 160 MB, without images or model calls):

```bash
uv sync --extra verify
uv run python scripts/data/audit_release.py \
  --cache rl_data/output/release-cache --out rl_data/output/release-checks.json
uv run python -m unittest discover -s tests -p 'test_verify*.py' -v
```

The released archive retains **710 conditional runtime dependency installers**.
Image baking reduced their need; it did not remove the fallback from those
scripts. Metadata flags **225 known verifier defects**, while the historical
prose reports 222. Preserve these limitations when describing the release.

## Provider configuration for new runs

Set `VERIFY_MODEL=gemini/<available-model>`, `openai/<available-model>`, or
`anthropic/<available-model>`. `VERIFY_ROLLOUT_MODEL` independently selects the
solver; every model CLI accepts `--model`. The restored client defaults to
Gemini Pro for audit/repair and Gemini Flash for rollouts. These are later
operational defaults, not the historical GPT 5.6 Sol judge / Terra solver.

Use `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), `OPENAI_API_KEY`, or
`ANTHROPIC_API_KEY`, matching the requested provider. Alternatively use
`<PROVIDER>_API_KEY_FILE` pointing to a mode-0600 file. The optional default
credential locations are `~/.config/tmax-evals/{gemini,openai,anthropic}-api-key`.
Gateway aliases are rejected. The private gateway client, service configuration,
and credentials are not distributed.

These two smoke commands make billable provider calls. Select available models
explicitly and run a small corpus pilot before scaling:

```bash
uv run python rl_data/verify/label_tasks.py --smoke --max-output-tokens 8192
uv run python rl_data/verify/smoke_tools.py --model "$VERIFY_ROLLOUT_MODEL"
```

The first exercises text and the rubric schema; the second checks a bash tool
call and continuation without executing the proposed command. Offline tests
mock provider responses. No new live provider test is claimed by this release.

## New corpus workflow

Run from the repository root with Python 3.12+. Image builds, extraction, sanity
checks, and rollouts need **Linux/AMD64, Apptainer with fakeroot, disk space for
the chosen images**, and the environment dependencies in `training/open-instruct/`.
Install that checkout's dependencies according to its README before running
sandboxes. Set `VERIFY_TRAINING_ROOT` for another checkout. Building these
AMD64 images on an ARM64 host is not covered by the tests.

Use absolute input paths and new output directories. `RUN`, `RECORDS`,
`TASK_DATA`, and `SIFS` below are your local locations; the original corpus is
always an input. `RECORDS` is Open Instruct JSONL with dict `env_config`, and
`TASK_DATA/<task_id>/` contains `instruction.md`, `setup.sh`, and `tests/test.sh`.
The index defaults to mangled registry-reference SIF filenames, documented by
`build_index.py --help` and `sif_name()`.

```bash
RUN="$PWD/rl_data/output/audit-run"
mkdir -p "$RUN"
uv run python rl_data/verify/build_index.py \
  --records "$RECORDS" --task-data "$TASK_DATA" --sifs "$SIFS" --out "$RUN/index.jsonl"
uv run python rl_data/verify/label_tasks.py \
  --index "$RUN/index.jsonl" --out "$RUN/labels.jsonl"
uv run python rl_data/verify/fix_tasks.py \
  --index "$RUN/index.jsonl" --labels "$RUN/labels.jsonl" --out "$RUN/fixes"
uv run python rl_data/verify/apply_repairs.py \
  --index "$RUN/index.jsonl" --fixes "$RUN/fixes" --mode extract
uv run python rl_data/verify/build_images.py \
  --index "$RUN/index.jsonl" --fixes "$RUN/fixes" --out "$RUN/images-initial" --apply
uv run python rl_data/verify/export_prefilter.py \
  --index "$RUN/index.jsonl" --labels "$RUN/labels.jsonl" --fixes "$RUN/fixes" \
  --plan-only --out-dir "$RUN/selection"
uv run python rl_data/verify/materialise.py \
  --index "$RUN/index.jsonl" --fixes "$RUN/fixes" \
  --tasks "$RUN/selection/candidate-task-ids.json" --src-task-data "$TASK_DATA" \
  --source-jsonl "$RECORDS" --image-builds "$RUN/images-initial" --out "$RUN/corpus-initial"
uv run python rl_data/verify/sanity.py --corpus "$RUN/corpus-initial" --out "$RUN/sanity"
```

`fix_tasks.py` proposes guarded verifier rewrites or removal of leaked paths.
Model-proposed `setup.sh` is retained in `fixes/` as provenance and is not executed.
`build_images.py` removes declared paths from a derivative of the original SIF.
Required oracle assets are extracted into tests and uploaded only at verification.
No `presetup.sh` hook is used. Extraction failures and unbuilt image changes stop
adoption. Refused repairs and their IDs stay in the exclusions table.
Resolve missing local SIFs before continuing after an extraction failure. An
accepted proposal whose required oracle could not be extracted is accounted for
as `oracle_extraction_incomplete` in selection exclusions, rather than adopted
without its asset.

The sanity pass runs each rewritten verifier on an untouched sandbox. A correct
rejection is expected; a crash, build failure, missing reward, or empty pass is
recorded separately. For missing verifier dependencies, `fix_runtime.py` can
print candidate modules but **its historical `--apply` is disabled**. Select
packages and exact versions deliberately, preserving the module/package mapping
(e.g. `yaml` is imported from `PyYAML`). An explicit dependency-plan JSON is:

```json
{
  "task_example": {
    "requirements": ["PyYAML==6.0.2"],
    "imports": ["yaml"]
  }
}
```

This is a format example, not a prescribed historical pin. Make the straightforward
runtime fixes revealed by that bounded pass, then build again **from the original
index**, combining leak removal and dependencies in one derivative:

```bash
uv run python rl_data/verify/build_images.py \
  --index "$RUN/index.jsonl" --fixes "$RUN/fixes" \
  --dependency-plan "$RUN/dependency-plan.json" --out "$RUN/images-final" --apply
uv run python rl_data/verify/materialise.py \
  --index "$RUN/index.jsonl" --fixes "$RUN/fixes" \
  --tasks "$RUN/selection/candidate-task-ids.json" --src-task-data "$TASK_DATA" \
  --source-jsonl "$RECORDS" --image-builds "$RUN/images-final" --out "$RUN/corpus-final"
uv run python rl_data/verify/sanity.py --corpus "$RUN/corpus-final" --out "$RUN/sanity"
```

If the initial sanity pass needs no fix, use `corpus-initial` as the final corpus.
The follow-up sanity command rechecks only changed artifacts in that same output
directory; completed results for unchanged tasks are retained.
Without `--apply`, the image command emits definitions and a plan only. With it,
every build and final image must pass all declared imports, requested version
checks, and leak-path absence checks. Receipts record source/image/recipe SHA256s;
`dep_pins.json` includes requested pins and versions of newly installed or changed
transitive packages read from the built image. A missing or modified receipt/image
cannot silently fall back to the original image during materialisation. Export
portable recipes with `make_recipe.py --builds "$RUN/images-final" --out <new-dir>`.
Registry tags and package indexes may change; receipts and existing images remain
the evidence for the actual build. A new image still needs final task validation.

Export the prefilter independently of solvability, or optionally run the historical
nominal two-attempt / 600-second / 40-step protocol with your chosen solver:

```bash
uv run python rl_data/verify/rollout.py --corpus "$RUN/corpus-final" \
  --out "$RUN/rollouts-final" --attempts 2 --deadline 600 --max-steps 40
uv run python rl_data/verify/export_verified.py --corpus "$RUN/corpus-final" \
  --rollouts "$RUN/rollouts-final" --index "$RUN/index.jsonl" \
  --labels "$RUN/labels.jsonl" --fixes "$RUN/fixes" --out-dir "$RUN/verified"
uv run python rl_data/verify/export_prefilter.py --corpus "$RUN/corpus-final" \
  --index "$RUN/index.jsonl" --labels "$RUN/labels.jsonl" --fixes "$RUN/fixes" \
  --verified "$RUN/verified/verified.jsonl" --sanity "$RUN/sanity" --out-dir "$RUN/export"
```

Omit `--verified` for a prefilter without rollout evidence. Final exports require
actual materialised patches. New evidence carries a fingerprint of every task file
including entry types, permissions and symlink targets, plus image bytes;
changed artifacts invalidate prior attempts regardless of directory
name. These checks read the SIFs and can take time on a full corpus. Historical
traces lack these fingerprints and cannot be admitted by the new export CLI.
Earlier content-only fingerprints also require fresh validation; relabeling old
traces with a new fingerprint would not establish what was executed.
Rewritten verifiers require a current sanity result. Known runtime defects remain
visible in the prefilter and are excluded from the demonstrated passing subset.
Sanity rechecks use fresh staging directories. Missing/invalid rewards and
infrastructure failures carry no scored evidence; verifier timeouts remain
separate budget outcomes rather than scored rejections.
Upload-root directories must be real directories. Supporting file symlinks are
preserved, and extracted oracle overlays replace links without writing through
them into the original corpus.
The episode deadline is checked between blocking steps, so 600 seconds is a
nominal budget, not a hard process kill guarantee.

`prefilter.jsonl` and `exclusions.jsonl` account for every original ID.
`verified-task-ids.json` is the demonstrated passing subset; use it with
`materialise.py` to create a separate postfilter corpus. This workflow neither
adds a second rubric audit nor requires a known-good replay gate. The restored
`gate.py`, `replay.py`, `export_hf.py` (including its alternative keep policy),
`export_repairs.py`, and `plot_labels.py` are historical analysis helpers with
legacy run-layout assumptions. They are not additional release gates.

## Publication conversion tools

See [`scripts/data/README.md`](../../scripts/data/README.md). Those scripts recover
the later SIF-to-Docker/Open Instruct publication route for the fixed v1.1 corpus;
they do not generate the judge labels or repairs. New exports from this workflow
are separate artifacts and need their own conversion/publication plan.
