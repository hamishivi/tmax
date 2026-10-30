"""Prepare, publish, and verify the evaluated 2B/27B preserved-thinking setup.

Requires huggingface_hub and jinja2. Preparation is read-only on the Hub.
Publication makes fast-forward Git commits with the prepared parent commits;
bare Git transfers LFS pointers, never model weight payloads.
"""

import argparse
import base64
import copy
import hashlib
import json
import os
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, get_token
from jinja2 import Environment

ROOT = Path(__file__).resolve().parent
CONTROLLED = {
    "README.md",
    "release_manifest.json",
    "chat_template.jinja",
    "preserved_thinking.json",
}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def read(repo, revision, name):
    with urllib.request.urlopen(
        f"https://huggingface.co/{repo}/resolve/{revision}/{name}", timeout=60
    ) as response:
        return response.read()


def inventory(info):
    return {
        f.rfilename: {
            "blob_id": f.blob_id,
            "size": f.size,
            "lfs": dict(f.lfs) if f.lfs else None,
        }
        for f in info.siblings
        if f.rfilename not in CONTROLLED
    }


def fingerprint(files):
    weights = {
        name: f["lfs"]["sha256"]
        for name, f in files.items()
        if name.endswith(".safetensors")
    }
    assert weights, "No model weights in source inventory"
    return hashlib.sha256(json.dumps(weights, sort_keys=True).encode()).hexdigest()


def validate_template(text):
    def fail(message):
        raise ValueError(message)

    template = Environment().from_string(text)
    history = [
        {"role": "user", "content": "Solve the task."},
        {
            "role": "assistant",
            "content": "Run a check.",
            "reasoning_content": "FIRST_REASONING",
        },
        {"role": "tool", "content": "Check output."},
    ]
    options = {"tools": [], "add_generation_prompt": True, "raise_exception": fail}
    for messages in (
        history,
        history + [{"role": "user", "content": "Please use the bash tool."}],
    ):
        rendered = template.render(messages=messages, **options)
        assert "FIRST_REASONING" in rendered
        assert rendered.endswith("<|im_start|>assistant\n<think>\n")
    history[1] = {
        "role": "assistant",
        "content": "<think>CONTENT_REASONING</think>Run a check.",
    }
    assert "CONTENT_REASONING" in template.render(messages=history, **options)
    disabled = template.render(messages=history, enable_thinking=False, **options)
    assert disabled.endswith("<think>\n\n</think>\n\n")


def patch_card(card, model, step):
    winner = model["winner"]
    card, count = re.subn(
        r"This branch contains \*\*step \d+\*\*\. The `main` branch contains \*\*step \d+\*\*",
        f"This branch contains **step {step}**. The `main` branch contains **step {winner}**",
        card,
    )
    assert count == 1
    start = card.index("| Training step | TB-Lite (%) ± SE (pp) | Selection |")
    end = card.index("\n\n", start)
    table = ["| Training step | TB-Lite (%) ± SE (pp) | Selection |", "|---|---:|---|"]
    for score in model["scores"]["tblite"]:
        n = score["step"]
        mark = " **main**" if n == winner else ""
        table.append(
            f"| [{n}](https://huggingface.co/{model['repo']}/tree/step-{n:04d}) | {score['score']:.2f} ± {score['stderr']:.2f} |{mark} |"
        )
    card = card[:start] + "\n".join(table) + card[end:]
    tb21 = model["scores"]["tb21"][0]
    tb20 = model["scores"]["tb20"][0]
    sentence = f"The selected checkpoint scores **{tb21['score']:.2f} ± {tb21['stderr']:.2f}%** on Terminal-Bench 2.1 (88 tasks, three trials per task; `configure-git-webserver` excluded)."
    sentence += f"\n\nOn Terminal-Bench 2.0 it scores **{tb20['score']:.2f} ± {tb20['stderr']:.2f}%** (all 89 tasks, three trials per task, Daytona backend)."
    card, count = re.subn(r"The selected checkpoint scores \*\*[^\n]+", sentence, card)
    assert count == 1
    paragraph = (
        "These scores use Vanillux 0.2.3 with preserved intermediate thinking and the TB-Lite checkpoint sweep completed in October 2026. "
        "The shipped `chat_template.jinja` exactly matches the template used for these evaluations: thinking is enabled by default, and previous reasoning is retained across tool responses and user reprompts. "
        "No chat-template override is required. An explicit `enable_thinking=false` can disable thinking. "
        "TB-Lite and Terminal-Bench 2.1 use Sandfleet through Harbor; Terminal-Bench 2.0 uses Daytona. "
        "All scores report three trials per task and standard errors in percentage points. "
        "`release_manifest.json` records the preserved-thinking sweep, and `preserved_thinking.json` records the exact template hash."
    )
    card, count = re.subn(r"These v1\.1 scores use [^\n]+", paragraph, card)
    assert count == 1
    card = re.sub(
        r"- \*\*Released training step:\*\* \d+",
        f"- **Released training step:** {step}",
        card,
    )
    card = re.sub(
        r"- \*\*Main checkpoint:\*\* \d+, selected on TB-Lite",
        f"- **Main checkpoint:** {winner}, selected on TB-Lite",
        card,
    )
    source = f"The source checkpoint is [{model['repo']} at step {step}](https://huggingface.co/{model['repo']}/tree/step-{step:04d})."
    card, count = re.subn(
        r"The source checkpoint is [^\n]+?\. Original checkpoint files are preserved",
        source + " Model weights and training provenance are preserved",
        card,
    )
    assert count == 1
    return card


def prepare(directory):
    api = HfApi()
    config = json.loads((ROOT / "preserved_thinking_scores.json").read_text())
    plan = {"models": [], "changes": []}
    for model in config["models"]:
        repo = model["repo"]
        template = (ROOT / model["template"]).read_bytes()
        assert hashlib.sha256(template).hexdigest() == model["template_sha256"]
        validate_template(template.decode())
        branches = [f"step-{n:04d}" for n in (100, 200, 300, 400, 500)] + ["main"]
        infos = {
            branch: api.model_info(repo, revision=branch, files_metadata=True)
            for branch in branches
        }
        old_manifest = json.loads(
            read(repo, infos["main"].sha, "release_manifest.json")
        )
        manifest = copy.deepcopy(old_manifest)
        manifest["previous_evaluation_without_preservation_override"] = {
            "winner": old_manifest["winner"],
            "tblite": old_manifest["tblite"],
            "tb21": old_manifest["tb21"],
        }
        manifest.update(model["scores"])
        manifest["winner"] = model["winner"]
        manifest["evaluation_chat_template_sha256"] = model["template_sha256"]
        manifest["evaluation_configuration"] = (
            "preserved intermediate thinking; enabled by default"
        )
        plan["models"].append({"repo": repo, "manifest": manifest})
        for branch in branches:
            before = infos[branch]
            step = (
                model["winner"]
                if branch == "main"
                else int(branch.removeprefix("step-"))
            )
            source = infos[f"step-{step:04d}"] if branch == "main" else before
            target = directory / repo.split("/")[-1] / branch
            target.mkdir(parents=True, exist_ok=True)
            card = patch_card(read(repo, before.sha, "README.md").decode(), model, step)
            (target / "README.md").write_text(card)
            (target / "chat_template.jinja").write_bytes(template)
            dump(target / "release_manifest.json", manifest)
            dump(
                target / "preserved_thinking.json",
                {
                    "checkpoint_step": step,
                    "selected_main_step": model["winner"],
                    "template_sha256": model["template_sha256"],
                    "template_matches_evaluation": True,
                    "thinking_enabled_by_default": True,
                    "retains_reasoning_across_user_reprompts": True,
                    "source_checkpoint_revision": source.sha,
                    "weight_shard_map_sha256": fingerprint(inventory(source)),
                },
            )
            plan["changes"].append(
                {
                    "repo": repo,
                    "branch": branch,
                    "parent": before.sha,
                    "source": source.sha,
                    "directory": str(target.relative_to(directory)),
                    "expected_artifacts": inventory(source),
                }
            )
        print(f"Prepared {repo}: six branches, main step {model['winner']}", flush=True)
    dump(directory / "plan.json", plan)


def verify_change(api, directory, change, revision):
    info = api.model_info(change["repo"], revision=revision, files_metadata=True)
    assert (
        inventory(info) == change["expected_artifacts"]
    ), f"Model artifacts differ: {change['repo']}@{revision}"
    for name in CONTROLLED:
        expected = (directory / change["directory"] / name).read_bytes()
        assert (
            read(change["repo"], revision, name) == expected
        ), f"Published {name} differs"
    return {
        "repo": change["repo"],
        "branch": change["branch"],
        "parent": change["parent"],
        "revision": revision,
        "source_revision": change["source"],
        "unchanged_artifacts_verified": len(inventory(info)),
        "weight_shard_map_sha256": fingerprint(inventory(info)),
    }


def apply(directory):
    api = HfApi(token=True)
    assert api.whoami()["name"] == "hamishivi"
    token = get_token()
    assert token
    header = (
        "Authorization: Basic "
        + base64.b64encode(("hamishivi:" + token).encode()).decode()
    )
    environment = os.environ.copy()
    environment.update(
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0="http.extraHeader",
        GIT_CONFIG_VALUE_0=header,
        GIT_LFS_SKIP_SMUDGE="1",
        GIT_AUTHOR_NAME="Hamish Ivison",
        GIT_AUTHOR_EMAIL="hamishivi@users.noreply.huggingface.co",
        GIT_COMMITTER_NAME="Hamish Ivison",
        GIT_COMMITTER_EMAIL="hamishivi@users.noreply.huggingface.co",
    )
    plan = json.loads((directory / "plan.json").read_text())
    receipts_path = directory / "receipts.json"
    receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else []
    for model in plan["models"]:
        changes = [c for c in plan["changes"] if c["repo"] == model["repo"]]
        with tempfile.TemporaryDirectory(prefix="tmax-template-pointers-") as checkout:

            def git(*args, data=None):
                result = subprocess.run(
                    ["git", *args],
                    cwd=checkout,
                    env=environment,
                    input=data,
                    capture_output=True,
                )
                if result.returncode:
                    error = (
                        result.stderr.decode(errors="replace")
                        .replace(token, "[REDACTED]")
                        .replace(header, "[REDACTED]")
                    )
                    raise RuntimeError(error)
                return result.stdout.decode().strip()

            git("init", "--bare", "--quiet")
            git("remote", "add", "origin", "https://huggingface.co/" + model["repo"])
            commits = sorted({c[k] for c in changes for k in ("parent", "source")})
            git("fetch", "--quiet", "--no-tags", "origin", *commits)
            for change in changes:
                current = api.model_info(change["repo"], revision=change["branch"]).sha
                existing = next(
                    (
                        r
                        for r in receipts
                        if r["repo"] == change["repo"]
                        and r["branch"] == change["branch"]
                    ),
                    None,
                )
                if current != change["parent"]:
                    assert (
                        existing and current == existing["revision"]
                    ), "Branch changed since preparation; prepare again"
                    verify_change(api, directory, change, current)
                    continue
                git("read-tree", change["source"])
                for name in sorted(CONTROLLED):
                    blob = git(
                        "hash-object",
                        "-w",
                        "--stdin",
                        data=(directory / change["directory"] / name).read_bytes(),
                    )
                    git("update-index", "--add", "--cacheinfo", f"100644,{blob},{name}")
                tree = git("write-tree")
                commit = git(
                    "commit-tree",
                    tree,
                    "-p",
                    change["parent"],
                    data=b"Align shipped template and checkpoint selection with preserved-thinking evaluations\n",
                )
                git(
                    "push",
                    "--quiet",
                    "origin",
                    f"{commit}:refs/heads/{change['branch']}",
                )
                receipt = verify_change(api, directory, change, commit)
                receipts.append(receipt)
                dump(receipts_path, receipts)
                print(
                    f"Published and verified {change['repo']}@{change['branch']}: {commit}",
                    flush=True,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "apply", "verify"))
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        prepare(args.directory)
    elif args.mode == "apply":
        apply(args.directory)
    else:
        api = HfApi()
        plan = json.loads((args.directory / "plan.json").read_text())
        for change in plan["changes"]:
            verify_change(api, args.directory, change, change["branch"])
        print(f"Verified all {len(plan['changes'])} public branches")


if __name__ == "__main__":
    main()
