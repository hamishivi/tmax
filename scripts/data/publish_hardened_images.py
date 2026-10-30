#!/usr/bin/env python3
"""Publish portable hardened TMax images without rebuilding pinned dependencies.

Deletion-only repairs append OCI whiteouts to the original Docker layers. Other
repairs convert the *published, validated* SIF root filesystem to a Docker layer,
retaining the original Docker runtime configuration. No task code is executed.
Credentials are read from a mode-0600 JSON file, never printed.
"""

import argparse
import concurrent.futures
import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
import tempfile
import threading
import time
from urllib.parse import urljoin

import requests

REGISTRY = "https://registry-1.docker.io"
# This historical revision belongs to the original source repository.
SOURCE = "wAI-org/swerl-tmax-15k-hardened-prefilter"
SOURCE_REVISION = "6f62db2f3ed0b1ebf386de2009a3d649d13e5616"
POOL = "TMaxxx/tmax-image-pool"
POOL_REVISION = "0c4e432c59f9fb7daa7c58dcc43bf31a75230cc8"
MANIFEST_TYPE = "application/vnd.docker.distribution.manifest.v2+json"
LAYER_TYPE = "application/vnd.docker.image.rootfs.diff.tar.gzip"
CONFIG_TYPE = "application/vnd.docker.container.image.v1+json"
ACCEPT = ", ".join([MANIFEST_TYPE, "application/vnd.oci.image.manifest.v1+json",
                    "application/vnd.docker.distribution.manifest.list.v2+json",
                    "application/vnd.oci.image.index.v1+json"])


def json_bytes(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def sha256(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


class Registry:
    def __init__(self, credentials, target):
        self.credentials = credentials
        self.target = target
        self.session = requests.Session()
        self.tokens = {}

    def token(self, repo, source=None, refresh=False):
        key = (repo, source)
        cached = self.tokens.get(key)
        if cached and cached[0] > time.time() and not refresh:
            return cached[1]
        scopes = ["repository:%s:pull,push" % repo]
        if source and source != repo:
            scopes.append("repository:%s:pull" % source)
        response = self.session.get(
            "https://auth.docker.io/token",
            params=[("service", "registry.docker.io")] + [("scope", s) for s in scopes],
            auth=(self.credentials["Username"], self.credentials["Secret"]), timeout=60)
        response.raise_for_status()
        result = response.json()
        token = result["token"]
        self.tokens[key] = (time.time() + min(result.get("expires_in", 300) - 30, 180), token)
        return token

    def request(self, method, path, repo, source=None, **kwargs):
        url = urljoin(REGISTRY, path)
        caller_headers = kwargs.pop("headers", {})
        data = kwargs.get("data")
        start = data.tell() if hasattr(data, "tell") else None
        for attempt in range(6):
            if start is not None:
                data.seek(start)
            headers = dict(caller_headers, Authorization="Bearer " + self.token(repo, source, attempt > 0))
            response = self.session.request(method, url, headers=headers, timeout=(30, 600), **kwargs)
            if response.status_code not in (401, 429, 500, 502, 503, 504):
                return response
            response.close()
            time.sleep(min(2 ** attempt, 30))
        raise RuntimeError("Registry %s %s failed: HTTP %s" % (method, repo, response.status_code))

    def manifest(self, repo, tag):
        response = self.request("GET", "/v2/%s/manifests/%s" % (repo, tag), repo,
                                headers={"Accept": ACCEPT})
        response.raise_for_status()
        manifest = response.json()
        if "manifests" in manifest:
            choices = [m for m in manifest["manifests"]
                       if m.get("platform", {}).get("os") == "linux"
                       and m.get("platform", {}).get("architecture") == "amd64"]
            if len(choices) != 1:
                raise ValueError("Expected exactly one linux/amd64 manifest for " + repo)
            return self.manifest(repo, choices[0]["digest"])
        return manifest, response.headers.get("Docker-Content-Digest")

    def has_blob(self, digest):
        response = self.request("HEAD", "/v2/%s/blobs/%s" % (self.target, digest), self.target)
        if response.status_code not in (200, 404):
            response.raise_for_status()
        return response.status_code == 200

    def upload(self, data, digest):
        if self.has_blob(digest):
            return
        response = self.request("POST", "/v2/%s/blobs/uploads/" % self.target, self.target)
        response.raise_for_status()
        location = response.headers["Location"]
        location += ("&" if "?" in location else "?") + "digest=" + digest
        response = self.request("PUT", location, self.target, data=data,
                                headers={"Content-Type": "application/octet-stream"})
        response.raise_for_status()
        if response.status_code != 201:
            raise RuntimeError("Blob upload did not finish")

    def mount(self, source, descriptor):
        digest = descriptor["digest"]
        if self.has_blob(digest):
            return
        response = self.request("POST", "/v2/%s/blobs/uploads/" % self.target,
                                self.target, source=source, params={"mount": digest, "from": source})
        response.raise_for_status()
        if response.status_code == 201:
            return
        # Fall back if Docker Hub cannot mount a public cross-repository blob.
        location = response.headers.get("Location")
        if location:
            self.request("DELETE", location, self.target).close()
        response = self.request("GET", "/v2/%s/blobs/%s" % (source, digest), source, stream=True)
        response.raise_for_status()
        with tempfile.TemporaryFile() as downloaded:
            check = hashlib.sha256()
            for block in response.iter_content(4 * 1024 * 1024):
                downloaded.write(block)
                check.update(block)
            if "sha256:" + check.hexdigest() != digest:
                raise ValueError("Base layer checksum mismatch")
            downloaded.seek(0)
            self.upload(downloaded, digest)

    def publish(self, tag, config, layers):
        raw = json_bytes(config)
        config_digest = sha256(raw)
        self.upload(raw, config_digest)
        manifest = {"schemaVersion": 2, "mediaType": MANIFEST_TYPE,
                    "config": {"mediaType": CONFIG_TYPE, "size": len(raw), "digest": config_digest},
                    "layers": layers}
        body = json_bytes(manifest)
        response = self.request("PUT", "/v2/%s/manifests/%s" % (self.target, tag), self.target,
                                data=body, headers={"Content-Type": MANIFEST_TYPE})
        response.raise_for_status()
        digest = sha256(body)
        if response.headers.get("Docker-Content-Digest") != digest:
            raise ValueError("Published manifest digest mismatch")
        response = self.request("HEAD", "/v2/%s/manifests/%s" % (self.target, tag), self.target,
                                headers={"Accept": ACCEPT})
        response.raise_for_status()
        if response.headers.get("Docker-Content-Digest") != digest:
            raise ValueError("Published tag does not resolve to expected digest")
        return digest


def leaked_paths(definition):
    paths = []
    for line in definition.splitlines():
        if line.strip().startswith("rm -rf "):
            tokens = shlex.split(line.strip())
            for path in tokens[2:]:
                if not path.startswith("/") or path == "/" or ".." in PurePosixPath(path).parts:
                    raise ValueError("Invalid leak path: " + path)
                if any(c in path for c in ["$", "*", "?", "`"]):
                    raise ValueError("Unsupported shell expression in leak path: " + path)
                paths.append(path)
    if not paths:
        raise ValueError("No removal paths in remove_leak recipe")
    return sorted(set(paths))


def whiteout_layer(paths):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in paths:
            name = PurePosixPath(path.lstrip("/"))
            entry = tarfile.TarInfo(str(name.parent / (".wh." + name.name)))
            entry.mode = 0
            entry.mtime = 0
            archive.addfile(entry)
    tar = stream.getvalue()
    compressed = gzip.compress(tar, compresslevel=3, mtime=0)
    return compressed, sha256(tar)


def download_sif(task, destination, expected):
    url = "https://huggingface.co/datasets/%s/resolve/%s/derived/%s.sif" % (POOL, POOL_REVISION, task)
    for attempt in range(5):
        check = hashlib.sha256()
        try:
            with requests.get(url, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                with destination.open("wb") as out:
                    for block in response.iter_content(4 * 1024 * 1024):
                        out.write(block)
                        check.update(block)
            if destination.stat().st_size != expected["size"]:
                raise ValueError("SIF size mismatch")
            if expected.get("sha256") and check.hexdigest() != expected["sha256"]:
                raise ValueError("SIF checksum mismatch")
            return check.hexdigest()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def convert_sif(sif, directory):
    listing = subprocess.check_output(["apptainer", "sif", "list", str(sif)], text=True)
    partition = [line.split("|")[0].strip() for line in listing.splitlines()
                 if "FS" in line and "Squashfs" in line]
    if len(partition) != 1 or not partition[0].isdigit():
        raise ValueError("Could not identify unique SIF Squashfs partition: " + listing)
    squash = directory / "root.sqfs"
    with squash.open("wb") as out:
        subprocess.run(["apptainer", "sif", "dump", partition[0], str(sif)], stdout=out, check=True)
    layer = directory / "root.tar.gz"
    errors = directory / "sqfs2tar.log"
    uncompressed = hashlib.sha256()
    with errors.open("wb") as err, layer.open("wb") as out:
        process = subprocess.Popen(["sqfs2tar", "--no-skip", "--root-becomes", ".", str(squash)],
                                   stdout=subprocess.PIPE, stderr=err)
        try:
            with gzip.GzipFile(fileobj=out, mode="wb", compresslevel=3, mtime=0, filename="") as zipped:
                while True:
                    block = process.stdout.read(4 * 1024 * 1024)
                    if not block:
                        break
                    uncompressed.update(block)
                    zipped.write(block)
            code = process.wait()
            if code:
                raise RuntimeError("sqfs2tar failed: " + errors.read_text()[-2000:])
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()
    compressed = hashlib.sha256()
    with layer.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            compressed.update(block)
    return layer, "sha256:" + compressed.hexdigest(), "sha256:" + uncompressed.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--credentials")
    ap.add_argument("--apply", action="store_true", help="Publish images; default prints the plan only")
    ap.add_argument("--output", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--kind", choices=["all", "whiteout", "sif"], default="all")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    if not args.apply:
        print(json.dumps({"repository": plan["repository"], "images": len(plan["images"]), "apply": False}))
        return 0
    if not args.credentials:
        ap.error("--credentials is required with --apply")
    credential_path = Path(args.credentials)
    if credential_path.stat().st_mode & 0o077:
        raise ValueError("Credentials file must have mode 0600")
    credentials = json.loads(credential_path.read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    Path(args.work).mkdir(parents=True, exist_ok=True)
    tasks = [t for t in plan["images"] if args.kind == "all" or t["kind"] == args.kind]
    if args.limit:
        tasks = tasks[:args.limit]
    def one(task):
        tid = task["task_id"]
        result_path = output / (tid + ".json")
        if result_path.exists():
            prior = json.loads(result_path.read_text())
            if prior.get("image") == task["image"] and prior.get("status") == "ok":
                return prior
        registry = Registry(credentials, plan["repository"])
        source_repo, source_tag = task["base_image"].rsplit(":", 1)
        base_manifest, base_digest = registry.manifest(source_repo, source_tag)
        response = registry.request("GET", "/v2/%s/blobs/%s" % (source_repo, base_manifest["config"]["digest"]), source_repo)
        response.raise_for_status()
        base_config = response.json()
        if base_config["architecture"] != "amd64" or base_config["os"] != "linux":
            raise ValueError("Unexpected base image platform")
        config = copy.deepcopy(base_config)
        config.pop("container", None)
        config.pop("container_config", None)
        config.pop("docker_version", None)
        labels = config.setdefault("config", {}).setdefault("Labels", {}) or {}
        config["config"]["Labels"] = labels
        labels.update({"org.opencontainers.image.source": "https://huggingface.co/datasets/" + SOURCE,
                       "tmax.task_id": tid, "tmax.source_revision": SOURCE_REVISION,
                       "tmax.derivation": ",".join(task["derivation"]),
                       "tmax.base_digest": base_digest or ""})
        started = time.time()
        with tempfile.TemporaryDirectory(prefix=tid + "-", dir=args.work) as temporary:
            temporary = Path(temporary)
            if task["kind"] == "whiteout":
                layer, diff_id = whiteout_layer(task["removed_paths"])
                layer_digest = sha256(layer)
                for descriptor in base_manifest["layers"]:
                    registry.mount(source_repo, descriptor)
                registry.upload(layer, layer_digest)
                layers = [{"mediaType": LAYER_TYPE, "size": d["size"], "digest": d["digest"]}
                          for d in base_manifest["layers"]]
                layers.append({"mediaType": LAYER_TYPE, "size": len(layer), "digest": layer_digest})
                config["rootfs"]["diff_ids"].append(diff_id)
                config.setdefault("history", []).append({"created_by": "TMax hardening: remove leaked answer assets"})
                extra = {"removed_paths": task["removed_paths"], "uploaded_layer_bytes": len(layer)}
            else:
                sif = temporary / "source.sif"
                checksum = download_sif(tid, sif, task["sif"])
                layer, layer_digest, diff_id = convert_sif(sif, temporary)
                with layer.open("rb") as blob:
                    registry.upload(blob, layer_digest)
                layers = [{"mediaType": LAYER_TYPE, "size": layer.stat().st_size, "digest": layer_digest}]
                config["rootfs"] = {"type": "layers", "diff_ids": [diff_id]}
                config["history"] = [{"created_by": "Import validated TMax SIF " + checksum}]
                labels["tmax.sif_sha256"] = checksum
                labels["tmax.image_pool_revision"] = POOL_REVISION
                extra = {"sif_sha256": checksum, "uploaded_layer_bytes": layer.stat().st_size}
            digest = registry.publish(task["image"].rsplit(":", 1)[1], config, layers)
        result = dict(task_id=tid, image=task["image"], status="ok", digest=digest,
                      kind=task["kind"], base_image=task["base_image"], base_digest=base_digest,
                      seconds=round(time.time() - started, 2), **extra)
        pending = result_path.with_suffix(".%s.%s.tmp" % (os.getpid(), threading.get_ident()))
        pending.write_text(json.dumps(result, indent=2) + "\n")
        pending.replace(result_path)
        return result

    errors = []
    print("Publishing %s images to %s with %s workers" % (len(tasks), plan["repository"], args.jobs), flush=True)
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as executor:
        futures = {executor.submit(one, t): t for t in tasks}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            task = futures[future]
            try:
                result = future.result()
                print("[%s/%s] OK %s %s %ss" % (i, len(tasks), task["task_id"], result["kind"], result.get("seconds")), flush=True)
            except Exception as exc:
                # Do not print requests' exception strings, which may contain signed URLs.
                message = "%s: %s" % (type(exc).__name__, str(exc)[:500]) if not isinstance(exc, requests.RequestException) else type(exc).__name__
                errors.append({"task_id": task["task_id"], "error": message})
                print("[%s/%s] FAILED %s %s" % (i, len(tasks), task["task_id"], message), flush=True)
    (output / ("errors-" + args.kind + ".json")).write_text(json.dumps(errors, indent=2) + "\n")
    print("Completed %s; failed %s" % (len(tasks) - len(errors), len(errors)), flush=True)
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
