# Running harbor evals on Beaker against a local vLLM

End-to-end: a single Beaker task that allocates N GPUs, serves a HuggingFace
model with vLLM on `localhost:8008`, brings up podman + harbor, and runs a
harbor dataset (terminal-bench or tblite) against it. Results land on weka.

The pipeline exists because harbor's `--env docker` path doesn't work
out-of-the-box on a podman socket — there are several rough edges around
networking, user namespaces, and bind-mount permissions that the patches in
this directory paper over. See [Why the patches?](#why-the-patches) at the
bottom if you're curious.

## Quickstart

### Running on Beaker

```bash
./beaker_configs/launch_eval.sh allenai/open_instruct_dev \
    --revision sft_qwen3_4b_tmax_4node \
    --name sft-4b \
    --dataset terminal-bench@2.0
```

Submits a single 8-GPU task. By default it uses `VanilluxAgent` against
`terminal-bench@2.0`. Results land at:

```
/weka/oe-adapt-default/$USER/tmax-eval/<job-name>/jobs/<job-name>/
```

### Running locally on a node with podman already installed

```bash
source scripts/setup_podman_harbor.sh
uv run harbor run \
    --dataset terminal-bench@2.0 \
    --agent oracle \
    --env docker
```

That sourced script does the minimum runtime fixes (mknod `/dev/net/tun`,
mkdir for `aardvark-dns`, start the podman socket, export `DOCKER_HOST`). It
does **not** apply the harbor source patches — those live in the package
files and need to be reapplied after `uv sync` (see
[harbor source patches](#harbor-source-patches)).

## What the Beaker job does

A single task running:

1. `apt-get install podman crun uidmap fuse-overlayfs slirp4netns netavark aardvark-dns`
2. Writes `/etc/containers/containers.conf` with `userns="auto:size=65536"`,
   `netns="host"`, `ipcns="host"`, `crun` runtime, cgroups disabled.
3. `git clone` the tmax repo at the SHA you launched from (the wrapper reads
   your current HEAD and origin URL).
4. `uv sync` then patches harbor's vendored files in `.venv/...`:
   - `harbor/environments/docker/docker-compose-base.yaml`: add
     `network_mode: host` and `:U` on the bind mounts.
   - `harbor/agents/oracle.py` and `harbor/verifier/verifier.py`: chmod the
     pre-touched files to 0666 and their parent dirs to 0777 so the
     user-namespaced container can write through the bind mount.
5. Runs `scripts/setup_podman_harbor.sh`: `mknod /dev/net/tun`,
   `mkdir -p /run/containers/networks/aardvark-dns`, starts
   `podman system service --time=0 unix:///tmp/podman.sock`, exports
   `DOCKER_HOST`.
6. Launches vLLM in the background:
   ```
   uvx --from vllm==0.19.1 vllm serve $MODEL_PATH --revision $MODEL_REVISION
                --served-model-name $SERVED_MODEL_NAME
                --enable-auto-tool-choice --tool-call-parser hermes
                --tensor-parallel-size $TP_SIZE
                --data-parallel-size $DP_SIZE
                --port $VLLM_PORT
   ```
   Logs go to `/tmp/vllm.log`.
7. Polls `http://localhost:$VLLM_PORT/v1/models` for up to 30 minutes. If
   vLLM dies before becoming ready, prints the tail of the log and exits 1.
8. Runs `uv run harbor run --env docker --agent-import-path
   $AGENT_IMPORT_PATH --model hosted_vllm/$SERVED_MODEL_NAME --dataset
   $DATASET --agent-kwarg api_base=http://localhost:$VLLM_PORT/v1`.
9. Runs `scripts/compute_stats.py jobs/$JOB_NAME` and writes the output to
   `jobs/$JOB_NAME/stats.txt` plus structured metrics to
   `jobs/$JOB_NAME/metrics.json`.
10. Copies `jobs/$JOB_NAME/` to `$RESULTS_DIR` on weka.

## Flags

```
./beaker_configs/launch_eval.sh <model_path> [options]
```

Required: `<model_path>` — HF identifier (or a weka path the beaker image can
read).

| Flag | Default | Notes |
|---|---|---|
| `--revision REV` | `main` | Passed as `--revision` and `--tokenizer-revision` to vLLM. |
| `--name NAME` | `basename(model_path)` | `--served-model-name` for vLLM. Also drives `JOB_NAME`. |
| `--gpus N` | `8` | |
| `--tp N` | `gpus` | Tensor-parallel size. |
| `--dp N` | `1` | Data-parallel size. |
| `--port PORT` | `8008` | vLLM port. |
| `--max-model-len LEN` | unset | Pass to `vllm serve --max-model-len`. |
| `--dataset DS` | `terminal-bench@2.0` | Also valid: `openthoughts-tblite@2.0`, `terminal-bench-pro@1.0`, `terminal-bench-sample@2.0`. |
| `--agent IMPORT_PATH` | `VanilluxAgent:VanilluxAgent` | `module:Class` form. |
| `--n-concurrent N` | `8` | Parallel trials. |
| `--n-attempts N` | `1` | Passed as `-k`. |
| `--job-name NAME` | `<served-name>-<dataset-slug>` | Harbor job name. Also the subdir under `RESULTS_DIR`. |
| `--results-dir DIR` | `/weka/oe-adapt-default/$USER/tmax-eval/<job-name>` | Destination on weka. |
| `--cluster CLUSTER` | `ai2/saturn` | |
| `--budget BUDGET` | `ai2/oe-adapt` | |
| `--priority PRI` | `high` | |
| `--workspace WS` | `$BEAKER_WORKSPACE` or `ai2/tmax` | |
| `--image IMAGE` | `ai2/cuda12.8-dev-ubuntu22.04-torch2.10.0` | Beaker image. |
| `--repo-url URL` | current `origin` URL | git URL the beaker task clones. |
| `--repo-ref REF` | current `git rev-parse HEAD` | git ref to check out. **Must be pushed.** |

## Prerequisites

- **HF_TOKEN beaker secret.** The yaml binds the `HF_TOKEN` beaker secret as
  an env var so vLLM can pull from gated HF repos.
- **weka access.** The yaml mounts the `oe-adapt-default` weka source at
  `/weka/oe-adapt-default`. Adjust `--results-dir` if your workspace uses a
  different mount.
- **Pushed commit.** The wrapper grabs your current HEAD SHA and embeds it
  in the yaml. If that SHA isn't on the remote, the beaker task can't clone
  it. The wrapper warns when this is the case — push first, or pass
  `--repo-ref` explicitly.

## Where to look when something goes wrong

Inside the beaker task, in order of when things fail:

| Symptom | Where to look |
|---|---|
| Beaker task exits before any "===" log line | Beaker task log — likely apt-install or git clone. |
| `unknown shorthand flag: 'p' in -p` from `docker compose down` | The image's `docker` CLI has no compose plugin. The script auto-installs Docker Compose v2 as a CLI plugin; if you see this anyway, check that `curl https://github.com/...` is reachable from your beaker network. |
| `mknod` permission denied | The cluster doesn't grant CAP_MKNOD. Ask your beaker admin or use an image with `/dev/net/tun` pre-created. |
| podman socket never appears | `/tmp/podman-service.log` for the daemon, then beaker task log. |
| vLLM never becomes ready | `/tmp/vllm.log` (the inner script tails it on failure). |
| Trials all `RewardFileNotFoundError` | Harbor patches didn't apply. Confirm the patched files via `python3 -c "import harbor, pathlib; print((pathlib.Path(harbor.__file__).parent/'agents/oracle.py').read_text())" \| grep chmod`. |
| Trials fail with `setgroups 65534` | Your podman's userns is too small. The script writes `auto:size=65536`; confirm `/etc/containers/containers.conf` actually has that line. |
| Reward `0` across the board | Look at one trial: `jobs/$JOB_NAME/<task>/agent/oracle.txt` (agent stdout) and `verifier/test-stdout.txt` (test output). Likely a real model/agent issue, not infra. |

After the job ends successfully:

- Aggregate: `/weka/.../tmax-eval/<job-name>/jobs/<job-name>/result.json` —
  reward distribution + exception stats.
- Stats summary: `/weka/.../tmax-eval/<job-name>/jobs/<job-name>/stats.txt` —
  mean reward, standard deviation, SEM, and pass@k.
- Metrics JSON: `/weka/.../tmax-eval/<job-name>/metrics.json` and
  `/weka/.../tmax-eval/<job-name>/jobs/<job-name>/metrics.json` — structured
  mean reward, standard deviation, SEM, run scores, pass@k, and per-task stats.
- Per-trial: `/weka/.../tmax-eval/<job-name>/jobs/<job-name>/<task>__<rand>/`
  with `agent/oracle.txt`, `verifier/test-stdout.txt`, `verifier/reward.txt`,
  `exception.txt`, `trial.log`.

## Why the patches?

Discovered while bringing harbor up against a rootful podman socket. Each
fix is in response to a concrete failure mode:

1. **`/dev/net/tun` missing.** Without it, pasta/netavark can't set up
   container networking — fails at `up --detach --wait` with
   `Failed to open() /dev/net/tun`. Fix: `mknod /dev/net/tun c 10 200`.
2. **`/run/containers/networks/aardvark-dns/` missing.** First netavark run
   fails with `failed to create aardvark-dns directory`. Fix: `mkdir -p`.
3. **Compose creates per-project networks.** With rootful podman in a
   sandboxed container, project networks were flaky. Fix: add
   `network_mode: host` to harbor's compose so containers share the host
   netns directly. Combined with `netns="host"` in containers.conf this
   skips podman's network plumbing entirely.
4. **Container can't write to bind-mounted host files.** With
   `userns="auto"`, container root maps to a host subuid (10000+); the
   pre-touched `oracle.txt` (host root, 644) can't be overwritten. Fix:
   chmod the pre-touched files 0666 and parent dirs 0777. (We tried `:U` on
   the bind mounts; podman-compose silently drops it. `:idmap` needs
   filesystem support that `/tmp` doesn't have.)
5. **`_apt` user (uid 65534) unmapped.** With the default
   `userns="auto:size=1024"`, uid 65534 isn't inside the namespace and
   `apt-get` in the container fails with `setgroups 65534`. Fix: bump to
   `auto:size=65536`.

The compose + harbor-source patches are reapplied on every beaker job
(idempotently, via sentinel checks). The local `scripts/setup_podman_harbor.sh`
only handles the runtime bits — patches need to be reapplied manually
locally after `uv sync`.

## Harbor source patches

If you're running locally, after `uv sync` reapply these:

- `.venv/lib/python3.12/site-packages/harbor/environments/docker/docker-compose-base.yaml`:
  add `network_mode: host` to the `main` service, append `:U` to each bind mount.
- `.venv/lib/python3.12/site-packages/harbor/agents/oracle.py`: after
  `host_oracle_path.touch()` add `host_oracle_path.chmod(0o666)` and
  `host_oracle_path.parent.chmod(0o777)`.
- `.venv/lib/python3.12/site-packages/harbor/verifier/verifier.py`: same
  treatment after `self._trial_paths.test_stdout_path.touch()`.

The inner beaker script (`scripts/beaker/run_eval_in_job.sh`) has these
edits as a Python heredoc you can rip out if you want to script them
locally.
