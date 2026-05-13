#!/usr/bin/env bash
#
# Inner script invoked inside a beaker task. Sets up podman + harbor, spins up
# a vLLM server on the local GPUs, then runs harbor against it.
#
# Driven entirely by env vars (set by beaker_configs/launch_eval.sh):
#   MODEL_PATH               HF model path or weka path (required)
#   MODEL_REVISION           HF revision/branch (default: main)
#   SERVED_MODEL_NAME        --served-model-name for vLLM (required)
#   VLLM_VERSION             vLLM package version for uvx (default: 0.19.1)
#   VLLM_PORT                port for vLLM (default: 8008)
#   TP_SIZE                  --tensor-parallel-size (required)
#   DP_SIZE                  --data-parallel-size (default: 1)
#   MAX_MODEL_LEN            optional --max-model-len
#   DATASET                  harbor dataset, e.g. terminal-bench@2.0
#   AGENT_IMPORT_PATH        e.g. VanilluxAgent:VanilluxAgent
#   N_CONCURRENT             default 8
#   N_ATTEMPTS               default 1
#   JOB_NAME                 harbor job name
#   RESULTS_DIR              /weka path to copy jobs/$JOB_NAME into
#   REPO_GIT_URL, REPO_GIT_REF
#                            optional — if set, this script self-clones into a
#                            workdir; otherwise it assumes pwd is the repo.

set -euo pipefail

log() { printf '\n=== [%s] %s ===\n' "$(date -u +%H:%M:%S)" "$*"; }

# --- 0. Workdir: clone repo if URL given, else use cwd ----------------------
if [ -n "${REPO_GIT_URL:-}" ]; then
    WORKDIR="${WORKDIR:-/workspace/tmax}"
    if [ ! -d "$WORKDIR/.git" ]; then
        log "cloning $REPO_GIT_URL @ ${REPO_GIT_REF:-HEAD} -> $WORKDIR"
        mkdir -p "$(dirname "$WORKDIR")"
        git clone "$REPO_GIT_URL" "$WORKDIR"
        if [ -n "${REPO_GIT_REF:-}" ]; then
            git -C "$WORKDIR" checkout "$REPO_GIT_REF"
        fi
    fi
    cd "$WORKDIR"
fi

# --- 1. Install podman + deps -----------------------------------------------
if ! command -v podman >/dev/null 2>&1; then
    log "installing podman + helpers"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq podman crun uidmap fuse-overlayfs slirp4netns \
        curl git ca-certificates
fi

# Some images (e.g. AI2's cuda gantry images) ship a real Docker CLI without
# the compose plugin. Harbor shells out to `docker compose ...`, so without
# the plugin every up/down errors with "unknown shorthand flag: 'p' in -p"
# (Docker CLI rejects `compose` as a subcommand and then mis-parses `-p`).
# Drop in the official compose v2 static binary as a user-level CLI plugin —
# it talks the Docker API, which podman serves on /tmp/podman.sock.
if ! docker compose version >/dev/null 2>&1; then
    log "installing docker compose v2 plugin"
    DOCKER_COMPOSE_VERSION="${DOCKER_COMPOSE_VERSION:-v2.39.4}"
    DOCKER_COMPOSE_ARCH="$(uname -m)"
    mkdir -p /root/.docker/cli-plugins
    curl -fsSL \
        "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-linux-${DOCKER_COMPOSE_ARCH}" \
        -o /root/.docker/cli-plugins/docker-compose
    chmod +x /root/.docker/cli-plugins/docker-compose
fi

# --- 2. Write containers.conf -----------------------------------------------
log "writing /etc/containers/containers.conf"
mkdir -p /etc/containers
cat > /etc/containers/containers.conf <<'CONF'
[containers]
netns="host"
userns="auto:size=65536"
ipcns="host"
utsns="host"
cgroupns="host"
cgroups="disabled"
keyring=false
log_driver = "k8s-file"
volumes = [
        "/proc:/proc",
]
default_sysctls = []
[engine]
cgroup_manager = "cgroupfs"
events_logger="file"
runtime="crun"
CONF

# Ensure root has a subuid/subgid range big enough for the userns size above.
grep -q '^root:' /etc/subuid 2>/dev/null || echo 'root:10000:65536' >> /etc/subuid
grep -q '^root:' /etc/subgid 2>/dev/null || echo 'root:10000:65536' >> /etc/subgid

# --- 3. Patch harbor's vendored compose + touch sites -----------------------
# Re-applied on every run; sentinel checks make this idempotent. Required
# because harbor is reinstalled fresh in each beaker container.
log "running uv sync"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv sync

log "patching harbor for podman compat"
uv run python - <<'PY'
import pathlib, harbor
hdir = pathlib.Path(harbor.__file__).parent

compose = hdir / "environments/docker/docker-compose-base.yaml"
text = compose.read_text()
if "network_mode: host" not in text:
    text = text.replace(
        "  main:\n    volumes:",
        "  main:\n    network_mode: host\n    volumes:",
    )
    for host, env in (
        ("HOST_VERIFIER_LOGS_PATH", "ENV_VERIFIER_LOGS_PATH"),
        ("HOST_AGENT_LOGS_PATH", "ENV_AGENT_LOGS_PATH"),
        ("HOST_ARTIFACTS_PATH", "ENV_ARTIFACTS_PATH"),
    ):
        text = text.replace(
            f"${{{host}}}:${{{env}}}",
            f"${{{host}}}:${{{env}}}:U",
        )
    compose.write_text(text)
    print("patched docker-compose-base.yaml")

oracle = hdir / "agents/oracle.py"
text = oracle.read_text()
if "host_oracle_path.chmod(0o666)" not in text:
    text = text.replace(
        "if environment.is_mounted:\n            host_oracle_path.touch()",
        "if environment.is_mounted:\n"
        "            host_oracle_path.touch()\n"
        "            host_oracle_path.chmod(0o666)\n"
        "            host_oracle_path.parent.chmod(0o777)",
    )
    oracle.write_text(text)
    print("patched oracle.py")

verifier = hdir / "verifier/verifier.py"
text = verifier.read_text()
if "test_stdout_path.chmod(0o666)" not in text:
    text = text.replace(
        "self._trial_paths.test_stdout_path.touch()",
        "self._trial_paths.test_stdout_path.touch()\n"
        "        self._trial_paths.test_stdout_path.chmod(0o666)\n"
        "        self._trial_paths.test_stdout_path.parent.chmod(0o777)",
    )
    verifier.write_text(text)
    print("patched verifier.py")

# Make agent_dir / verifier_dir / artifacts_dir world-writable on the host so
# user-namespaced container writes (anything that doesn't go through a
# pre-touched harbor file: SWE-agent's *.traj, swe-agent.txt, etc.) don't
# silently fail with permission-denied on the bind mount.
paths_py = hdir / "models/trial/paths.py"
text = paths_py.read_text()
if "agent_dir.chmod(0o777)" not in text:
    text = text.replace(
        "self.agent_dir.mkdir(parents=True, exist_ok=True)\n"
        "        self.verifier_dir.mkdir(parents=True, exist_ok=True)\n"
        "        self.artifacts_dir.mkdir(parents=True, exist_ok=True)",
        "self.agent_dir.mkdir(parents=True, exist_ok=True)\n"
        "        self.verifier_dir.mkdir(parents=True, exist_ok=True)\n"
        "        self.artifacts_dir.mkdir(parents=True, exist_ok=True)\n"
        "        self.agent_dir.chmod(0o777)\n"
        "        self.verifier_dir.chmod(0o777)\n"
        "        self.artifacts_dir.chmod(0o777)",
    )
    paths_py.write_text(text)
    print("patched paths.py")

# Drop --rmi all from compose-down: harbor deletes the image after every
# trial, which makes each retry of a tb2 task re-pull from Docker Hub and
# blows past the unauthenticated pull cap. Keeping images on local podman
# storage costs ~9 GB total for tb2 (89 unique images) but eliminates the
# re-pull storm entirely.
docker_py = hdir / "environments/docker/docker.py"
text = docker_py.read_text()
if '["down", "--rmi", "all", "--volumes", "--remove-orphans"]' in text:
    text = text.replace(
        '["down", "--rmi", "all", "--volumes", "--remove-orphans"]',
        '["down", "--volumes", "--remove-orphans"]',
    )
    docker_py.write_text(text)
    print("patched docker.py: dropped --rmi all from compose down")
PY

# --- 4. Bring podman service up (uses scripts/setup_podman_harbor.sh) -------
log "starting podman service"
# shellcheck disable=SC1091
source scripts/setup_podman_harbor.sh
export DOCKER_HOST="${DOCKER_HOST:-unix:///tmp/podman.sock}"

# --- 4a. Docker Hub auth + mirror -------------------------------------------
# tb2 task images live on Docker Hub; on a 267-trial run, harbor's
# `compose down --rmi all` deletes each image after a trial, so the next
# trial re-pulls and we blow past Docker Hub's 100 pulls/6hr unauthenticated
# cap somewhere around trial 100 (whole 2nd half of the run fails with
# "toomanyrequests: You have reached your unauthenticated pull rate limit").
# Defense in depth:
#   - if the image ships an internal Docker Hub mirror, use it
#   - if DOCKER_PAT is set (beaker secret), write the auth config (200/6hr
#     authenticated cap, or unlimited on a Docker Hub paid account)
#   - --rmi all is dropped via the docker.py patch below (step 3 already
#     ran, but the sed below is idempotent and safe to re-run)
if [ -x /usr/local/bin/setup_dockerio_mirror ]; then
    /usr/local/bin/setup_dockerio_mirror || log "setup_dockerio_mirror failed (continuing)"
fi
if [ -n "${DOCKER_PAT:-}" ]; then
    log "writing Docker Hub credentials"
    DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-hamishi740}"
    python3 - <<PY
import base64, json, os
username = "$DOCKERHUB_USERNAME"
pat = os.environ["DOCKER_PAT"]
auth = base64.b64encode(f"{username}:{pat}".encode()).decode()
cfg_dir = os.path.expanduser("~/.docker")
os.makedirs(cfg_dir, exist_ok=True)
cfg_path = os.path.join(cfg_dir, "config.json")
cfg = {}
if os.path.exists(cfg_path):
    try:
        cfg = json.load(open(cfg_path))
    except Exception:
        cfg = {}
cfg.setdefault("auths", {})["https://index.docker.io/v1/"] = {"auth": auth}
json.dump(cfg, open(cfg_path, "w"), indent=2)
print(f"wrote {cfg_path}")
PY
else
    log "DOCKER_PAT not set — Docker Hub pulls will be rate-limited (100/6hr)"
fi

# --- 5. Start vLLM in the background ----------------------------------------
: "${VLLM_VERSION:=0.19.1}"
: "${VLLM_PORT:=8008}"
: "${DP_SIZE:=1}"
VLLM_LOG=/tmp/vllm.log
VLLM_LOG_TAIL_LINES="${VLLM_LOG_TAIL_LINES:-300}"
VLLM_CMD=( uvx "vllm==${VLLM_VERSION}" serve "$MODEL_PATH"
           --revision "$MODEL_REVISION"
           --tokenizer-revision "$MODEL_REVISION"
           --served-model-name "$SERVED_MODEL_NAME"
           --enable-auto-tool-choice
           --tool-call-parser hermes
           --port "$VLLM_PORT"
           --gpu-memory-utilization 0.85
           --tensor-parallel-size "$TP_SIZE"
           --data-parallel-size "$DP_SIZE" )
if [ -n "${MAX_MODEL_LEN:-}" ]; then
    VLLM_CMD+=( --max-model-len "$MAX_MODEL_LEN" )
fi

log "launching vllm: ${VLLM_CMD[*]}"
"${VLLM_CMD[@]}" >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
    log "cleanup: killing vllm pid $VLLM_PID"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

log "waiting for vllm on :$VLLM_PORT (up to 30 min)"
for _ in $(seq 1 360); do
    if curl -sf "http://localhost:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
        log "vllm ready"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        log "vllm process died — tail of $VLLM_LOG:"
        tail -"$VLLM_LOG_TAIL_LINES" "$VLLM_LOG" || true
        exit 1
    fi
    sleep 5
done

if ! curl -sf "http://localhost:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
    log "vllm did not become ready in 30 min — tail of $VLLM_LOG:"
    tail -"$VLLM_LOG_TAIL_LINES" "$VLLM_LOG" || true
    exit 1
fi

# --- 6. Run harbor ----------------------------------------------------------
: "${N_CONCURRENT:=8}"
: "${N_ATTEMPTS:=1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export OPENAI_API_BASE="http://localhost:$VLLM_PORT/v1"
# Harbor's SWE-agent adapter copies OPENAI_BASE_URL (litellm convention) —
# not OPENAI_API_BASE — into the container, and only then does it pass
# --agent.model.api_base=... to sweagent. Without this, litellm in the
# container falls back to https://api.openai.com and every trial exits
# with NotFoundError: Hosted_vllmException on step 1.
export OPENAI_BASE_URL="http://localhost:$VLLM_PORT/v1"

HARBOR_CMD=( uv run harbor run
             --dataset "$DATASET"
             --agent-import-path "$AGENT_IMPORT_PATH"
             --model "hosted_vllm/$SERVED_MODEL_NAME"
             --env docker
             --n-concurrent "$N_CONCURRENT"
             --agent-kwarg "api_base=http://localhost:$VLLM_PORT/v1"
             --job-name "$JOB_NAME"
             -k "$N_ATTEMPTS" )
log "running harbor: ${HARBOR_CMD[*]}"

# Background progress reporter — harbor's built-in progress bar uses ANSI
# escapes that gantry logs flatten into noise, so we tail result.json
# ourselves and emit one human-readable line per interval.
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-60}"
RESULT_JSON="jobs/$JOB_NAME/result.json"
(
    while true; do
        sleep "$PROGRESS_INTERVAL"
        [ -f "$RESULT_JSON" ] || continue
        python3 - "$RESULT_JSON" <<'PY' || true
import json, sys, datetime
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
total = d.get("n_total_trials", "?")
stats = d.get("stats", {}) or {}
done = stats.get("n_trials", 0)
errs = stats.get("n_errors", 0)
evals = stats.get("evals", {}) or {}
mean = 0.0
for v in evals.values():
    m = (v.get("metrics") or [{}])[0].get("mean")
    if m is not None:
        mean = m
        break
ts = datetime.datetime.utcnow().strftime("%H:%M:%S")
print(f"=== [{ts}] progress: {done}/{total} trials  errors={errs}  mean={mean:.3f} ===",
      flush=True)
PY
    done
) &
PROGRESS_PID=$!

set +e
"${HARBOR_CMD[@]}"
HARBOR_RC=$?
set -e

kill "$PROGRESS_PID" 2>/dev/null || true
wait "$PROGRESS_PID" 2>/dev/null || true

# --- 7. Compute aggregate stats ---------------------------------------------
JOB_DIR="jobs/$JOB_NAME"
if [ -d "$JOB_DIR" ]; then
    STATS_LOG="$JOB_DIR/stats.txt"
    METRICS_JSON="$JOB_DIR/metrics.json"
    log "computing stats: scripts/compute_stats.py $JOB_DIR"
    set +e
    uv run python scripts/compute_stats.py "$JOB_DIR" --json-output "$METRICS_JSON" 2>&1 | tee "$STATS_LOG"
    STATS_RC=${PIPESTATUS[0]}
    set -e
    if [ "$STATS_RC" -ne 0 ]; then
        log "compute_stats.py exited with $STATS_RC; preserving harbor exit code $HARBOR_RC"
    else
        log "stats written to $STATS_LOG"
        log "metrics written to $METRICS_JSON"
    fi
else
    log "job directory $JOB_DIR not found; skipping compute_stats.py"
fi

# --- 8. Persist results to /weka --------------------------------------------
if [ -n "${RESULTS_DIR:-}" ]; then
    if [ -d "$JOB_DIR" ]; then
        log "copying $JOB_DIR -> $RESULTS_DIR/"
        mkdir -p "$RESULTS_DIR"
        cp -r "$JOB_DIR" "$RESULTS_DIR/"
        log "results available at $RESULTS_DIR/$JOB_NAME"
        if [ -f "$JOB_DIR/metrics.json" ]; then
            cp "$JOB_DIR/metrics.json" "$RESULTS_DIR/metrics.json"
            log "metrics available at $RESULTS_DIR/metrics.json"
        fi
    else
        log "job directory $JOB_DIR not found; skipping result copy"
    fi
fi

exit "$HARBOR_RC"
