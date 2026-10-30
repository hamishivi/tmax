#!/usr/bin/env python3
"""Prepare the hardened prefilter with the exact allenai Open Instruct schema."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

from publish_hardened_images import POOL, POOL_REVISION, SOURCE, SOURCE_REVISION, leaked_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--registry-tags", type=Path, required=True)
    parser.add_argument("--repository", default="hamishivi/swerl-tmax-hardened")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.source / "tasks.jsonl").open()]
    reference = pq.read_table(args.reference)
    manifests = {m["task_id"]: m for m in map(json.loads, (args.source / "image-recipe/manifest.jsonl").open())}
    tags = set(json.loads(args.registry_tags.read_text())["tags"])
    info = HfApi().dataset_info(POOL, revision=POOL_REVISION, files_metadata=True)
    sifs = {Path(f.rfilename).stem: {"size": f.size, "sha256": f.lfs.sha256}
            for f in info.siblings if f.rfilename.startswith("derived/")}
    images = []
    expected_ids = {row["ground_truth"] for row in rows}
    assert len(expected_ids) == len(rows) == 13565
    for row in rows:
        task_id = row["ground_truth"]
        assert task_id == row["env_config"]["task_id"]
        manifest = manifests[task_id]
        image = manifest["registry_image"]
        assert image.rsplit(":", 1)[1] in tags, image
        if manifest["needs_build"]:
            image = "%s:%s-%s" % (args.repository, task_id, POOL_REVISION[:8])
            kind = "whiteout" if manifest["derivation"] == ["remove_leak"] else "sif"
            record = {"task_id": task_id, "image": image, "base_image": manifest["registry_image"],
                      "derivation": manifest["derivation"], "kind": kind, "sif": sifs[task_id]}
            if "remove_leak" in manifest["derivation"]:
                record["removed_paths"] = leaked_paths((args.recipe / "image-recipe/defs" / (task_id + ".def")).read_text())
            images.append(record)
        row["env_config"]["image"] = image
        assert list(row) == reference.column_names
    assert len(images) == 1713
    # Struct and list element types, column order, and HF feature metadata match.
    table = pa.Table.from_pylist(rows, schema=reference.schema)
    assert table.schema == reference.schema
    assert table.num_rows == len(rows)
    verifiers, archive_ids = set(), set()
    with tarfile.open(args.source / "task-data.tar.gz", "r:gz") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            archive_ids.add(name.split("/", 1)[0])
            if name.endswith("/tests/test.sh") and member.isfile():
                verifiers.add(name.split("/", 1)[0])
    assert archive_ids == expected_ids, (len(archive_ids), len(expected_ids), sorted(archive_ids - expected_ids)[:5])
    assert verifiers == expected_ids
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "data").mkdir(exist_ok=True)
    pq.write_table(table, args.output / "data/train-00000-of-00001.parquet", compression="zstd")
    shutil.copy2(args.source / "task-data.tar.gz", args.output / "task-data.tar.gz")
    shutil.copy2(args.source / "prefilter.jsonl", args.output / "audit-metadata.jsonl")
    with (args.output / "tasks.jsonl").open("w") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    plan = {"source": SOURCE, "source_revision": SOURCE_REVISION, "image_pool": POOL,
            "image_pool_revision": POOL_REVISION, "repository": args.repository, "images": images}
    (args.output / "image-publication-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    report = {"rows": len(rows), "columns": table.column_names,
              "reference_schema_equal": table.schema == reference.schema,
              "archive_task_ids_equal": archive_ids == expected_ids,
              "verifiers": len(verifiers), "existing_registry_tasks": len(rows) - len(images),
              "images_to_publish": len(images), "image_kinds": dict(Counter(i["kind"] for i in images)),
              "source_revision": SOURCE_REVISION, "image_pool_revision": POOL_REVISION,
              "task_archive_sha256": hashlib.sha256((args.output / "task-data.tar.gz").read_bytes()).hexdigest()}
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
