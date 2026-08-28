#!/usr/bin/env python3
"""Exercise the TMAX Sandfleet backend against a real service pool."""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import os
import tarfile
import time
from dataclasses import asdict

from open_instruct.environments.backends import create_backend


def exercise(index: int, args: argparse.Namespace) -> dict:
    backend = create_backend(
        "sandfleet",
        image=args.image,
        timeout=args.command_timeout,
        mem_limit=args.mem_limit,
        sandfleet_url=args.url,
        sandfleet_pool=args.pool,
        sandfleet_token=args.token,
    )
    started_at = time.perf_counter()
    try:
        backend.start()
        cold_start_s = time.perf_counter() - started_at
        first_instance = backend._name
        first_cgroup = backend._worker_metadata.get("path")
        worker_pid = backend._sandfleet_status.get("worker_pid")

        command = backend.run_command(f"printf 'sandbox-{index}\\n'; python3 -c 'print(6 * 7)'")
        assert command.exit_code == 0, asdict(command)
        assert command.stdout == f"sandbox-{index}\n42\n", asdict(command)

        payload = f"remote-file-{index}\n".encode()
        backend.write_file("/workspace/sandfleet.txt", payload)
        assert backend.read_file("/workspace/sandfleet.txt", binary=True) == payload

        archive_buffer = io.BytesIO()
        with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
            info = tarfile.TarInfo("archive.txt")
            archive_payload = f"archive-{index}\n".encode()
            info.size = len(archive_payload)
            archive.addfile(info, io.BytesIO(archive_payload))
        backend.put_archive("/workspace", archive_buffer.getvalue())
        assert backend.read_file("/workspace/archive.txt") == f"archive-{index}\n"

        restarted_at = time.perf_counter()
        backend.restart()
        restart_s = time.perf_counter() - restarted_at
        second_instance = backend._name
        second_cgroup = backend._worker_metadata.get("path")
        second_worker_pid = backend._sandfleet_status.get("worker_pid")
        assert first_instance != second_instance
        assert first_cgroup == second_cgroup
        assert worker_pid == second_worker_pid
        clean = backend.run_command("test ! -e /workspace/sandfleet.txt && echo clean")
        assert clean.exit_code == 0 and clean.stdout == "clean\n", asdict(clean)
        return {
            "index": index,
            "cold_start_s": round(cold_start_s, 3),
            "restart_s": round(restart_s, 3),
            "first_instance": first_instance,
            "second_instance": second_instance,
            "cgroup": first_cgroup,
            "worker_pid": worker_pid,
        }
    finally:
        backend.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("SANDFLEET_URL"), required=False)
    parser.add_argument("--token", default=os.environ.get("SANDFLEET_CLIENT_TOKEN"), required=False)
    parser.add_argument("--pool")
    parser.add_argument("--mem-limit", default="4g")
    parser.add_argument("--image", required=True)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--command-timeout", type=int, default=120)
    args = parser.parse_args()
    if not args.url or not args.token:
        parser.error("--url/--token or SANDFLEET_URL/SANDFLEET_CLIENT_TOKEN are required")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(exercise, index, args) for index in range(args.concurrency)]
        results = [future.result() for future in futures]
    print(json.dumps({"ok": True, "sandboxes": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
