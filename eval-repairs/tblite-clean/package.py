"""Build a portable, audited task snapshot without changing source tasks."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import tomllib

OFFLINE_VERIFIER_ENV = {
    "UV_OFFLINE": "1",
    "PIP_NO_INDEX": "1",
    "PIP_FIND_LINKS": "/tests/_bootstrap/wheels",
    "UV_LINK_MODE": "copy",
}


def check_task_config(source, selected):
    before = tomllib.loads((source / "task.toml").read_text())
    after = tomllib.loads((selected / "task.toml").read_text())
    original_env = before.get("verifier", {}).get("env", {})
    final_env = after.get("verifier", {}).get("env", {})
    additions = {k: v for k, v in final_env.items() if k not in original_env}
    if any(final_env.get(k) != v for k, v in original_env.items()) or any(
        OFFLINE_VERIFIER_ENV.get(k) != v for k, v in additions.items()
    ):
        raise ValueError("Unapproved verifier environment change")
    for config in (before, after):
        if "verifier" in config:
            config["verifier"].pop("env", None)
    if before != after:
        raise ValueError("Task resources or configuration changed")


def hashes(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not supported: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def build(source, repairs, plan, output):
    if output.exists():
        raise FileExistsError(output)
    tasks = sorted(p.name for p in source.iterdir() if (p / "task.toml").is_file())
    if not tasks or set(tasks) != set(plan["tasks"]):
        raise ValueError("Plan must enumerate every source task exactly")
    if not plan["source_revision"]:
        raise ValueError("A pinned source revision is required")
    prepared = []
    excluded = {}
    for name in tasks:
        entry = plan["tasks"][name]
        original = hashes(source / name)
        if original != entry["source_files"]:
            raise ValueError(f"Source hash mismatch: {name}")
        if entry["status"] == "excluded":
            if not entry.get("reason"):
                raise ValueError(f"Missing exclusion reason: {name}")
            excluded[name] = entry["reason"]
            continue
        if entry["status"] not in ("unchanged", "validated-repair"):
            raise ValueError(f"Unvalidated task cannot enter release: {name}")
        selected = source / name
        if entry["status"] == "validated-repair":
            if not entry.get("evidence") or not entry.get("image_sha256"):
                raise ValueError(f"Missing validation/image evidence: {name}")
            selected = repairs / name
        final = hashes(selected)
        changes = {
            path: {"before": original.get(path), "after": final.get(path)}
            for path in sorted(original.keys() | final.keys())
            if original.get(path) != final.get(path)
        }
        if changes != entry.get("changes", {}):
            raise ValueError(f"Unreviewed task changes: {name}")
        # These are never altered by an offline/bootstrap repair.
        protected = {"instruction.md"} | {p for p in original if p.startswith("tests/") and p.endswith(".py")}
        if protected & changes.keys():
            raise ValueError(f"Instructions, resources or assertions changed: {name}")
        if "task.toml" in changes:
            check_task_config(source / name, selected)
        prepared.append((name, selected, final, entry))
    # All validation precedes writing. A partial copy has no release manifest.
    output.mkdir(parents=True)
    manifest = {
        "format": 1,
        "name": "tblite-clean",
        "source_revision": plan["source_revision"],
        "included_count": len(prepared),
        "excluded": excluded,
        "tasks": {},
        "note": "Unchanged means copied, not independently validated. Never pool different manifests silently.",
    }
    for name, selected, final, entry in prepared:
        shutil.copytree(selected, output / "tasks" / name)
        manifest["tasks"][name] = {**entry, "files": final}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repairs", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.repairs, json.loads(args.plan.read_text()), args.output)


if __name__ == "__main__":
    main()
