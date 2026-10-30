"""Replay saved npm manifests offline; dependency diagnostics, never model scores."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def replay(image, cache, trial):
    files = [trial / "package.json"]
    if not files[0].is_file():
        raise ValueError(f"Missing package.json in {trial}")
    files.extend(path for name in ("package-lock.json", "npm-shrinkwrap.json") if (path := trial / name).is_file())
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    # No agent scripts run, no registry access, no source or lockfile changes
    # propagate back to the input. npm may update its disposable working copy.
    command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "--network",
        "none",
        "--cpus",
        "2",
        "--memory",
        "4g",
        "--mount",
        f"type=bind,source={trial.resolve()},target=/input,readonly",
        "--mount",
        f"type=bind,source={cache.resolve()},target=/opt/npm-cache",
        "--entrypoint",
        "bash",
        image,
        "-c",
        (
            "set -eu; mkdir /tmp/replay; cd /tmp/replay; "
            "cp /input/package.json .; "
            "for name in package-lock.json npm-shrinkwrap.json; do "
            "if [ -f /input/$name ]; then cp /input/$name .; fi; done; "
            "timeout 120 npm install --offline --ignore-scripts --audit=false --fund=false "
            "--cache /opt/npm-cache"
        ),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "trial": trial.name,
        "input_sha256": hashes,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Pinned local image SHA256")
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument(
        "--trials", required=True, type=Path, help="Directory of trial-ID subdirectories containing saved manifests"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    image = subprocess.run(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], capture_output=True, text=True, check=True
    ).stdout.strip()
    trials = sorted(path for path in args.trials.iterdir() if path.is_dir())
    if not trials:
        raise ValueError("No trial directories supplied")
    with args.output.open("x") as output:
        for trial in trials:
            result = replay(image, args.cache, trial)
            result["image_sha256"] = image
            output.write(json.dumps(result) + "\n")
            output.flush()
            print(f"{trial.name}: install exit {result['exit_code']}", flush=True)


if __name__ == "__main__":
    main()
