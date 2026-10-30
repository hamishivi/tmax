# Provenance of the public task-audit and repair pipeline

## Answer to “is the untracked directory the code used for the paper?”

**It contains the historical audit/repair implementation, but it is not an
unchanged snapshot or a complete historical run package.** Eighteen Python
modules match the implementation on private branch `verify/tmax15k-rubric`,
commit `da201afad48cdb4826c0e759e1d111daeaa44b05` (September 13, 2026),
exactly or with identifiable later adaptations. The local `llm_client.py`
replaces the original gateway, and the original build-time dependency repair
orchestration was not committed on that branch. This public version reconstructs
that missing component and makes further adoption/export checks explicit.

This is source and artifact provenance, not proof that every recovered line was
executed in the historical job. The private source history and issues #12/#21
identify the workflow; the public artifacts supply reproducible evidence of its
outputs. The private threads themselves, gateway client/configuration, machine
paths, credentials, job records, and operational discussion are not distributed.

The investigation used public master
`584a6defbc26f110748c6c299cefbb2fc6885e89`, the local untracked sources,
the historical Git objects, the private decision/history threads, the supplied
Overleaf source snapshot, and pinned public Hugging Face artifacts. No paper or
training recipe was changed. The original working checkout was not modified.

## Source classification

The comparison below describes the imported local files against the historical
commit above. “Exact at import” refers to the untracked local source before
publication changes, not necessarily this PR's final file.

| Components | Relationship at import | Public preparation changes |
|---|---|---|
| `rubric.py` | Exact historical rubric/schema/prompts | Retained |
| `export_hf.py`, `export_repairs.py`, `plot_labels.py` | Exact historical analysis/export helpers | Retained; alternative policy and legacy assumptions documented |
| `fix_runtime.py` | Exact historical verifier-time pip installer | `--apply` disabled; retained for historical context and dry-run triage |
| `export_verified.py` | Exact historical solvability exporter | Final-artifact fingerprints replace run-directory naming as admission evidence; deduplicate attempts |
| `build_index.py` | Historical join/reconciliation, with root parameterised | Retained |
| `apply_repairs.py` | Historical oracle extraction/leak definitions, with protected roots parameterised | Validate removal paths/collisions; fail extraction on missing assets |
| `fix_tasks.py`, `label_tasks.py` | Historical prompt/guard/resumability flow with direct-client imports, provider-specific keys and smoke changes | Clear stale proposed scripts on a refused retry; no claim of historical model equivalence |
| `repair.py` | Historical prompts/guards; local docstring corrected to describe image removal | Validate all declared changes independently of action; reject unsafe paths, parent-directory removals affecting instruction inputs, and oracle basename collisions |
| `rollout.py` | Historical solver/harness with later provider defaults, proper tool-result messages, signature retention and tool-history trimming | Fingerprint final task/image artifacts, refuse stale resume evidence, distinguish invalid rewards/timeouts from scored rejections |
| `materialise.py` | Historical corpus adoption with protected roots parameterised | Fresh output only; require checked image builds; preserve supporting tests and symlinks; refuse incomplete repairs/assets |
| `export_prefilter.py` | Historical final keep policy with root parameterised | Explicit inputs, plan-only candidate IDs, final adoption checks, separate exclusions and validation status |
| `make_recipe.py` | Historical recipe generator with root parameterised | Export from checked build receipts and explicit dependency import names |
| `sanity.py` | Historical bounded empty-submission runtime check with root parameterised | Explicit final corpus; fresh staging with all test assets; fingerprint results and preserve verifier failure metadata |
| `gate.py`, `replay.py` | Historical A/B/replay analysis with roots parameterised | Optional historical analyses, not required release gates; the shared `run_arm` propagates missing-reward, timeout and infrastructure metadata |
| `llm_client.py`, `smoke_tools.py`, `__init__.py` | Later local additions; absent from the historical branch | Direct-provider replacement and provider tool smoke; not historical gateway code |
| `image_safety.py`, `build_images.py` | Absent from both historical branch and imported directory | New public reconstruction and shared validation |
| `scripts/data/prepare_hardened_open_instruct.py`, `publish_hardened_images.py` | Exact copies of the later scripts shipped with the public training dataset | Canonical public repository names; publication requires explicit `--apply` |
| `scripts/data/audit_release.py`, pipeline tests, provenance documents | New publication work | Independent artifact checks, offline regression coverage, limitations |

The historical commits progress from rubric/index/labeling (September 10), to
repair and image removal (September 11), to action/label decoupling and
solvability export (September 12), to runtime sanity and recipe generation
(September 13). The imported directory's October 3 README reports later direct
API smoke tests. Those historical smoke claims are not fresh tests performed by
this publication task.

## What matches the paper and released artifacts

The supplied paper snapshot's repair table reconciles with the released metadata:

| Original rubric label | Original tasks | Retained with applied repair |
|---|---:|---:|
| VERIFIER-TOO-WEAK | 11,160 | 10,990 |
| ANSWER-LEAK | 1,592 | 1,018 |
| INSTR-VERIFIER-MISMATCH | 441 | 432 |
| CLEAN | 1,125 | 0 (retained unchanged) |
| INSTR-ENV-MISMATCH | 177 | 0 |
| OTHER | 83 | 0 |
| INSTR-AMBIGUOUS | 23 | 0 |
| TASK-TRIVIAL | 0 | 0 |
| Total | 14,601 | 12,440 |

Thus **12,440 repaired + 1,125 clean = 13,565 retained**, with **1,036 separate
exclusions**. The final historical policy superseded an earlier request to keep
all 14,601 in the rewritten dataset. A refused/proposed repair does not count as
applied. Applied repairs do not imply a second rubric classification.

The training artifact is the hardened **prefilter**, not the solvability filter.
Its metadata has 7,015 `validated`, 6,520 `not_validated`, 26 `not_evaluable`, and
4 `budget_limited` tasks. The 7,015 demonstrated passing tasks are a separate
subset under the historical solver and nominal two-attempt, 600-second, 40-step
budget. Neither infrastructure failure nor failure to solve proves unsolvability.
See the public [training dataset](https://huggingface.co/datasets/TMaxxx/tmax-13k-v1.1/tree/84c6b85842e6ea108db97647ebbe48f9f1c8e7be)
and [metadata companion](https://huggingface.co/datasets/TMaxxx/tmax-13-v1.1-metadata/tree/62fdc43e34e8c8e9895aa3126ec02e1c1233591c).

The public conversion retains **11,852 original Docker references** and publishes
**1,713 hardened images**: 1,014 deletion-only whiteout images and 699 SIF imports
for dependency/combined repairs. The source recipe describes 1,763 derivatives
across all 14,601 original IDs; 1,713 belong to the retained 13,565. These are
different denominators, not conflicting counts. Image derivations must be keyed
by task ID: some tasks share a registry base tag but require different changes.
The source metadata/image-pool revisions are
`6f62db2f3ed0b1ebf386de2009a3d649d13e5616` and
`0c4e432c59f9fb7daa7c58dcc43bf31a75230cc8` respectively.

The original conversion files are independently downloadable at the pinned
[conversion directory](https://huggingface.co/datasets/TMaxxx/tmax-13k-v1.1/tree/84c6b85842e6ea108db97647ebbe48f9f1c8e7be/conversion).
Their SHA256s match the untracked local `scripts/data/` files at import. They
converted already repaired artifacts for publication; they did not produce the
paper's judge labels or repair proposals.

## Historical limitations and missing pieces

- The original judge and solver used a private provider gateway. A direct-provider
  model with a similar name is not evidence of identical model weights, routing,
  defaults, refusal policy, or verdicts. The public client is a replacement.
- The checked-in `fix_runtime.py` installs dependencies during verification.
  Later records report build-time installation and import checks, but the job
  orchestration that performed this and read per-image pins was not in the
  historical branch. Published `.def` recipes and validated SIFs preserve the
  outputs. `build_images.py` is a **new reconstruction**, requiring explicit
  per-task pinned requirements and import names; it does not recover historical
  version selection automatically.
  The artifact's `image-recipe/rebuild.sh` and `build_derived_local.py` are
  recoverable public reconstruction launchers, although absent from the local
  imported directory and historical branch. They rebuild from existing recipes
  or pins; they do not recover the missing original dependency-selection driver.
- The pinned task archive still contains the conditional `runtime-defect fix`
  bootstrap in **710 verifiers**. On images where the import works it skips pip;
  otherwise the fallback can still attempt a network install. Do not claim that
  the historical build-time transition removed this code from the released
  archive or that the archive alone proves network-free verification.
- The pinned metadata marks **225 known verifier defects**. The published audit
  prose reports 222 after dependency repair. Both are retained as evidence; this
  source publication does not resolve or silently rewrite that discrepancy.
- Full original launch commands, dependency-build logs, exact environment locks,
  and an artifact-to-run execution manifest are not in the imported directory.
  Sample evidence on the private branch is not a complete run record and is not
  republished here. Historical trace truncation and changed provider behavior
  also limit replay. The public metadata and recipes support output provenance,
  not byte-identical regeneration of all model outputs or images.
- New public materialisation/export interfaces are deliberately stricter than
  the historical scripts. They are documented in the pipeline README; old
  run-specific commands may need adaptation. New evidence fingerprints prevent
  stale admission but cannot retrospectively authenticate old traces.

## Checks performed for this publication

`python scripts/data/audit_release.py` checks pinned public revisions and writes
a local report to the requested `--out` path. It checks
all 13,565 training records and metadata rows, all 1,036 exclusions against the
original 14,601 IDs, the exact Open Instruct schema including HF metadata, required
archive files and instruction/prompt agreement, the 1,713 image publication
receipts/plan and 1,763-source-derivative accounting, and conversion-source hashes.
The task archive SHA256 is
`a654638ec448403823273b6510eeb457ae6d9914c7bf4f87ce5302d1fb0c7146`.

The 36 offline tests cover provider routing/retries/auth redaction, tool-call history,
repair guards, checked build adoption, resolved dependency pins, missing oracle
and image handling, supporting test preservation, keep/exclusion accounting,
stale validation rejection, and deterministic OCI whiteouts. Every Python source
compiles and all command-line help entry points load.

No corpus-scale generation, new provider call, actual Apptainer build/rollout,
registry upload, or Hugging Face write was performed for this PR. Build execution
is tested with controlled subprocess doubles and shell probes. A live
Linux/AMD64 pilot remains necessary before running the reconstructed driver at
scale. Existing images and task files remain the authoritative historical
artifacts; no identical historical reproducibility is claimed.
