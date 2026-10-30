#!/usr/bin/env python3
"""Reconstructed build-time repair orchestration; not the historical run driver.

Emit recipes from accepted repairs and an explicit, per-task dependency plan.
With --apply, build on Linux/AMD64 with Apptainer and validate the final image.
No verifier-time installs, inference of pip names, or automatic unpinned upgrades.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import re
import subprocess
import tempfile

try:
    from .image_safety import dependency_spec, removal_paths, sha256_file, task_id
except ImportError:
    from image_safety import dependency_spec, removal_paths, sha256_file, task_id


def check_script(paths: list[str], dependencies: dict | None) -> str:
    lines = ["set -eu"]
    for path in paths:
        quoted = shlex.quote(path)
        lines.extend([f"test ! -e {quoted}", f"test ! -L {quoted}"])
    if dependencies:
        payload = json.dumps(dependencies)
        code = ("import importlib, importlib.metadata, json; "
                f"p=json.loads({payload!r}); "
                "[importlib.import_module(m) for m in p['imports']]; "
                "[(importlib.metadata.version(r.split('==')[0]) == r.split('==')[1]) "
                "or (_ for _ in ()).throw(RuntimeError('Dependency version mismatch: '+r)) "
                "for r in p['requirements']]")
        lines.append("python3 -c " + shlex.quote(code))
    return "\n".join(lines) + "\n"


def recipe(base: str, paths: list[str], dependencies: dict | None) -> str:
    if "\n" in base or "\r" in base:
        raise ValueError("Image path cannot contain a newline")
    lines = ["Bootstrap: localimage", f"From: {base}", "", "%post", "    set -eu"]
    if paths:
        lines.append("    rm -rf -- " + " ".join(map(shlex.quote, paths)))
    if dependencies:
        lines.append("    python3 -m pip install --disable-pip-version-check --no-input "
                     + " ".join(map(shlex.quote, dependencies["requirements"])))
    lines.extend("    " + line for line in check_script(paths, dependencies).splitlines())
    return "\n".join(lines) + "\n"


def prepare(index: Path, fixes: Path, dependencies: dict) -> list[dict]:
    rows, seen = [], set()
    for line in index.read_text().splitlines():
        item = json.loads(line)
        tid = task_id(item["task_id"])
        if tid in seen:
            raise ValueError(f"Duplicate task: {tid}")
        seen.add(tid)
        repair_file = fixes / tid / "repair.json"
        repair = json.loads(repair_file.read_text()) if repair_file.exists() else {}
        paths = removal_paths(repair.get("leaked_paths", [])) if repair.get("accepted") else []
        if repair.get("accepted") and repair.get("needs_image_rebuild") and not paths:
            raise ValueError(f"{tid}: image rewrite has no supported removal paths")
        dep = dependency_spec(dependencies[tid]) if tid in dependencies else None
        rows.append({"task_id": tid, "registry_image": item["image"],
                     "base_sif": item.get("sif_path", ""), "removed_paths": paths,
                     "dependencies": dep,
                     "derivation": (["remove_leak"] if paths else []) + (["install_deps"] if dep else []),
                     "needs_build": bool(paths or dep)})
    if set(dependencies) - seen:
        raise ValueError("Dependency plan contains tasks absent from the index")
    return rows


def build_one(row: dict, definition: Path, output: Path, *, runner=subprocess.run) -> dict:
    """Publish a SIF only after build and final-image checks both succeed."""
    base = Path(row["base_sif"]).resolve()
    if not base.is_file() or base.stat().st_size == 0:
        raise ValueError(f"Missing base SIF for {row['task_id']}")
    source_hash = sha256_file(base)
    inventory_code = ("import importlib.metadata as m,json; "
                      "print(json.dumps({d.metadata['Name']: d.version for d in m.distributions()}))")
    def inventory(image):
        response = runner(['apptainer', 'exec', '--cleanenv', '--containall', '--pwd', '/',
                           str(image), 'python3', '-c', inventory_code],
                          check=True, capture_output=True, text=True, timeout=180)
        return {re.sub(r'[-_.]+', '-', name).lower(): version
                for name, version in json.loads(response.stdout).items()}
    before = inventory(base) if row['dependencies'] else {}
    with tempfile.TemporaryDirectory(prefix=".build-", dir=output) as temp:
        pending = Path(temp) / "image.sif"
        runner(["apptainer", "build", "--fakeroot", str(pending), str(definition)],
               check=True, capture_output=True, timeout=1800)
        runner(["apptainer", "exec", "--cleanenv", "--containall", "--pwd", "/",
                str(pending), "sh", "-c", check_script(row["removed_paths"], row["dependencies"])],
               check=True, capture_output=True, timeout=180)
        if not pending.is_file() or not pending.stat().st_size:
            raise ValueError("Build produced no image")
        if sha256_file(base) != source_hash:
            raise ValueError("Base image changed during the build")
        resolved = None
        if row['dependencies']:
            after = inventory(pending)
            pins = {re.sub(r'[-_.]+', '-', name).lower(): version
                    for name, version in (req.split('==', 1) for req in row['dependencies']['requirements'])}
            pins.update({name: version for name, version in after.items() if before.get(name) != version})
            resolved = dependency_spec({'requirements': [f'{name}=={version}' for name, version in sorted(pins.items())],
                                        'imports': row['dependencies']['imports']})
        receipt = {"task_id": row["task_id"], "status": "ok", "base_sha256": source_hash,
                   "recipe_sha256": sha256_file(definition), "image_sha256": sha256_file(pending),
                   "removed_paths": row["removed_paths"], "dependencies": row["dependencies"],
                   "resolved_dependencies": resolved}
        pending.replace(output / (row["task_id"] + ".sif"))
        return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--fixes", type=Path, required=True)
    parser.add_argument("--dependency-plan", type=Path,
                        help='JSON: {task_id: {"requirements": ["package==version"], "imports": ["module"]}}')
    parser.add_argument("--out", type=Path, required=True, help="New output directory; never overwrites a run")
    parser.add_argument("--apply", action="store_true", help="Build and check images (requires Linux/AMD64)")
    args = parser.parse_args()
    dependencies = json.loads(args.dependency_plan.read_text()) if args.dependency_plan else {}
    rows = prepare(args.index, args.fixes, dependencies)
    # Validate every base before creating any output.
    for row in rows:
        if row["needs_build"]:
            base = Path(row["base_sif"]).resolve() if row["base_sif"] else None
            if base is None or not base.is_file() or not base.stat().st_size:
                parser.error(f"Missing base SIF for {row['task_id']}")
            row["base_sif"] = str(base)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "defs").mkdir()
    (args.out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    receipts, pins = [], {}
    for row in rows:
        if not row["needs_build"]:
            continue
        definition = args.out / "defs" / (row["task_id"] + ".def")
        definition.write_text(recipe(row["base_sif"], row["removed_paths"], row["dependencies"]))
        if args.apply:
            try:
                receipt = build_one(row, definition.resolve(), args.out)
                if row["dependencies"]:
                    pins[row["task_id"]] = receipt['resolved_dependencies']['requirements']
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                # Build stderr can contain authenticated package-index URLs.
                receipt = {"task_id": row["task_id"], "status": "failed", "error": type(exc).__name__}
            receipts.append(receipt)
            (args.out / "builds.json").write_text(json.dumps(receipts, indent=2) + "\n")
            (args.out / "dep_pins.json").write_text(json.dumps(pins, indent=2) + "\n")
    if args.apply:
        (args.out / 'builds.json').write_text(json.dumps(receipts, indent=2) + '\n')
        (args.out / 'dep_pins.json').write_text(json.dumps(pins, indent=2) + '\n')
    print(json.dumps({"recipes": sum(r["needs_build"] for r in rows), "built": sum(r['status'] == 'ok' for r in receipts),
                      "applied": args.apply, "failed": sum(r["status"] != "ok" for r in receipts)}))
    return int(any(r["status"] != "ok" for r in receipts))


if __name__ == "__main__":
    raise SystemExit(main())
