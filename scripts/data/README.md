# TMax v1.1 publication conversion

`prepare_hardened_open_instruct.py` and `publish_hardened_images.py` were already
shipped in the public dataset's `conversion/` directory. Their imported local
copies matched those files byte for byte. They are later publication tooling,
separate from the historical rubric/repair run. See
[provenance and limitations](../../docs/task-audit-provenance.md).

`audit_release.py` downloads pinned public task records, metadata, recipes,
publication receipts, and the task archive. It verifies the schema against the
original Open Instruct export, compares all task IDs with the original dataset,
checks the task archive SHA256, checks instructions against training messages,
and reconciles image derivations. It does not execute task scripts or pull images.

```bash
uv sync --extra verify
uv run python scripts/data/audit_release.py \
  --cache rl_data/output/release-cache --out rl_data/output/release-checks.json
```

The command writes its report to the requested `--out` path under
`rl_data/output/`; generated reports are local run outputs. The report distinguishes
artifact checks from prior authors' runtime claims. Missing historical model
access and uncommitted build orchestration prevent an identical rerun of the paper.

The conversion scripts are **specific to the released 13,565-task corpus**:

1. Prepare a local snapshot of the source metadata revision
   `6f62db2f3ed0b1ebf386de2009a3d649d13e5616`, containing `tasks.jsonl`,
   `prefilter.jsonl`, `task-data.tar.gz`, and `image-recipe/`. The canonical public
   repository is now `TMaxxx/tmax-13-v1.1-metadata`.
2. Provide the original Open Instruct Parquet reference, recipe root (the directory
   containing `image-recipe/`), and an exported original Docker repository tag list
   as `{"tags": [...]}`. The script checks 13,565 unique tasks, 1,713 derived
   images, archive coverage, and reference schema. It writes to a local directory.
3. Review the resulting `image-publication-plan.json`. Deletion-only images use
   OCI whiteouts; dependency/combined repairs import the validated SIF from the
   pinned image pool. They do not rerun the proposed setup rewrite. The plan is
   not a generic converter for newly generated corpora.

```bash
uv run python scripts/data/prepare_hardened_open_instruct.py \
  --source "$SOURCE_SNAPSHOT" --recipe "$RECIPE_ROOT" --reference "$REFERENCE_PARQUET" \
  --registry-tags "$ORIGINAL_TAGS_JSON" --repository your-account/your-images \
  --output rl_data/output/publication-prepared
uv run python scripts/data/publish_hardened_images.py \
  --plan rl_data/output/publication-prepared/image-publication-plan.json \
  --output rl_data/output/image-receipts --work rl_data/output/image-staging
```

The second command prints the planned image count without publishing. To publish,
add `--apply --credentials <mode-0600-json-file>`; the file contains Docker Hub
`Username` and `Secret`. Linux tools `apptainer` and `sqfs2tar` are needed for SIF
conversion. `--limit 1 --jobs 1` selects a small publication pilot. Successful
per-task receipts permit resuming. Publishing creates registry tags; this PR's
checks did not invoke it. Private credentials and machine-specific run logs are
not included.
