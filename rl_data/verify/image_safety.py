"""Validation shared by the public image-build and materialisation tools."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import re
import json
import os
import stat


def task_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"task_[A-Za-z0-9_-]+", value):
        raise ValueError(f"Invalid task ID: {value!r}")
    return value


def removal_paths(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if (not isinstance(value, str) or not re.fullmatch(r"/[A-Za-z0-9_./@+-]+", value)
                or value == "/" or any(p in {".", ".."} for p in value.split("/"))
                or str(PurePosixPath(value)) != value):
            raise ValueError(f"Unsafe removal path: {value!r}")
        result.append(value)
    return sorted(set(result))


def dependency_spec(value: dict) -> dict:
    requirements = value.get("requirements", [])
    imports = value.get("imports", [])
    if not isinstance(requirements, list) or not isinstance(imports, list):
        raise ValueError("Dependency requirements and imports must be lists")
    if not requirements or not imports:
        raise ValueError("Each dependency plan needs exact requirements and explicit imports")
    for req in requirements:
        if not isinstance(req, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*==[A-Za-z0-9][A-Za-z0-9_.+!-]*", req):
            raise ValueError(f"Expected a pinned package==version: {req!r}")
    names = [re.sub(r"[-_.]+", "-", r.split("==")[0]).lower() for r in requirements]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate package in dependency plan")
    for mod in imports:
        if not isinstance(mod, str) or not re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*", mod):
            raise ValueError(f"Invalid import name: {mod!r}")
    return {"requirements": requirements, "imports": imports}


def sha256_file(path: Path) -> str:
    check = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            check.update(block)
    return check.hexdigest()


def corpus_fingerprints(corpus: Path) -> dict[str, str]:
    """Bind validation evidence to the image bytes and all final task files."""
    images, result = {}, {}
    for line in (corpus / 'tasks.jsonl').read_text().splitlines():
        row = json.loads(line)
        env = row['env_config']
        env = json.loads(env) if isinstance(env, str) else env
        tid = task_id(env['task_id'])
        if tid in result:
            raise ValueError(f'Duplicate corpus task: {tid}')
        image = Path(env['image']).resolve()
        if image not in images:
            images[image] = sha256_file(image)
        root = corpus / 'task-data' / tid
        for name in ('tests', 'environment', 'environment/seeds'):
            if (root / name).is_symlink():
                raise ValueError(f'Upload directories must be real directories: {tid}/{name}')
        files = {}
        for path in sorted(root.rglob('*')):
            entry_stat = path.lstat()
            entry = {'mode': stat.S_IMODE(entry_stat.st_mode)}
            if stat.S_ISLNK(entry_stat.st_mode):
                entry.update(type='symlink', target=os.readlink(path))
            elif stat.S_ISDIR(entry_stat.st_mode):
                entry.update(type='directory')
            elif stat.S_ISREG(entry_stat.st_mode):
                entry.update(type='file', sha256=sha256_file(path))
            else:
                raise ValueError(f'Unsupported filesystem entry: {path}')
            files[str(path.relative_to(root))] = entry
        if not {'instruction.md', 'setup.sh', 'tests/test.sh'}.issubset(files):
            raise ValueError(f'Incomplete task: {tid}')
        for name in ('instruction.md', 'setup.sh', 'tests/test.sh'):
            if files[name]['type'] != 'file':
                raise ValueError(f'Required task file must be regular: {tid}/{name}')
        payload = json.dumps({'image_sha256': images[image], 'files': files}, sort_keys=True)
        result[tid] = hashlib.sha256(payload.encode()).hexdigest()
    return result
