# Offline evaluation dependency repairs

These tools prepare separately identified dependency and bootstrap variants of
the pinned OpenThoughts-TBLite corpus. They preserve instructions, task
resources, reference solutions, Python assertions, and model-visible application
state. Keep the original corpus intact and record the prepared task/image hashes.

Use upstream revision `f075e463472c7790b85793b392dff1fff20cc0e3` from
[OpenThoughts-TBLite](https://github.com/open-thoughts/OpenThoughts-TBLite/tree/f075e463472c7790b85793b392dff1fff20cc0e3).

## Maven, MLflow, and OkHttp

The `exclusions-20260909/` directory provides dependency-only image recipes and
bootstrap preparation. Set `TBLITE_SOURCE` to a pristine copy of the pinned task
root, then run `prepare.py`, `prepare_maven.py`, and `extract_dependency_pom.py`
from that directory. Preparation creates new task directories and refuses to
overwrite existing ones.

Maven caches dependencies and its test provider without putting a repaired POM
or compiled application into the runtime image. MLflow preserves both phases
of the verifier's dependency installation in an offline cache. OkHttp caches
external runtime dependencies while retaining local application rebuilds.

`offline_guard.sh` captures output and preserves the verifier's exit status.
It does not infer infrastructure failure by matching arbitrary grader text,
which may be controlled by the model. Validate dependency provisioning before
model execution and retain ambiguous failures with their raw logs.

`tblite-clean/integrate_images.py` composes the dependency recipes into complete
task image contexts. The standalone Dockerfiles expect the corresponding
prepared upstream image tags as their base images.

## ETL, Hydra, and React

- `prepare_etl.py` serializes database initialization with verification and
  supports TCP endpoints without changing application logic or assertions.
- `prepare_hydra.py` stages a pinned source mirror and wheelhouse for the
  original dependency revision.
- `prepare_react.py` stages pinned Node/npm tools and a dependency-only cache.
  `capture_react_manifests.py`, `replay_react_cache.py`, and
  `react-cache-packages.txt` support reproducing the selected cache contents.

These are preparation tools, not a claim that every runtime supports every task.
PostgreSQL user switching needs valid user/group mappings; root-only mappings
are insufficient. Preserve CPU, memory, disk and timeout limits, and keep
provisioning failures separate from scored model outcomes.

## Portable task packaging

`tblite-clean/package.py` copies a complete source corpus and reviewed repairs
using an explicit release plan. Each task is unchanged, a validated repair, or
excluded with a reason. Source/repair hashes and exact changed paths are required.
The packager rejects source drift, unlisted changes, symlinks, and changes to
instructions, assertions or task resources. The only permitted task-config
additions are the defined offline verifier environment settings.

Provide portable evidence references and immutable image identities in a release
plan. Different task/image manifests describe different evaluation variants;
they should not silently replace historical scores.
