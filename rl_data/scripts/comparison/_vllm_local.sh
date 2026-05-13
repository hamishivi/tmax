#!/usr/bin/env bash
# In-job vLLM launcher, designed to be `source`d from the comparison
# run_generate_solutions_*.sh scripts so a single SLURM job can both serve
# the model AND run the solver against it.  No-op when LAUNCH_VLLM != 1, so
# sourcing this from the existing scripts is safe for the gemini/API path.
#
# Usage:
#   source rl_data/scripts/comparison/_vllm_local.sh
#   _vllm_start_local       # launches vLLM in background, installs cleanup trap.
#                           # Returns immediately so SIF builds can overlap.
#   ... slow stuff (SIF base build, per-task SIF builds) ...
#   _vllm_wait_ready_local  # block on /v1/models until 200 OK or timeout.
#                           # Sets HOSTED_VLLM_API_BASE / MODEL / OPENAI_API_KEY.
#
# When LAUNCH_VLLM != 1, both functions are no-ops.
#
# Knobs (all overridable via env, with sensible Qwen3-8B-on-8xH200 defaults):
#
#   LAUNCH_VLLM            1 to enable; anything else = no-op.  Default 0.
#   VLLM_MODEL             HF repo to serve.  Default Qwen/Qwen3-8B.
#   VLLM_PORT              Bind port.  Default: auto-pick a free port.
#   VLLM_HOST              Bind/connect host.  Default 127.0.0.1.
#   VLLM_TP                Tensor-parallel size.  Default 1.
#   VLLM_DP                Data-parallel size.  Default = visible GPU count
#                          (so an 8xH200 job becomes TP=1, DP=8 by default,
#                          which is the throughput-optimal config for an
#                          8B model that fits on one GPU).
#   VLLM_MAX_LEN           Max model context length.  Default 32768.
#   VLLM_GPU_UTIL          --gpu-memory-utilization.  Default 0.85.
#   VLLM_DTYPE             --dtype.  Default bfloat16.
#   VLLM_TOOL_CALL_PARSER  Tool-call parser.  Default hermes (works for
#                          Qwen2.5/Qwen3 OpenAI-compatible tool calling).
#                          Set empty to disable --enable-auto-tool-choice.
#   VLLM_REASONING_PARSER  Reasoning parser (split <think> from content).
#                          Default qwen3 when VLLM_MODEL contains "Qwen3";
#                          set to a literal empty string to disable.
#   VLLM_DISABLE_THINKING  1 to inject `chat_template_kwargs={enable_thinking:false}`
#                          via --chat-template-kwargs (faster turns; default 1
#                          for Qwen3 models, 0 otherwise).
#   VLLM_EXTRA_ARGS        Free-form extra args appended to the vllm command.
#   VLLM_LOG               Path for the vLLM server log.
#                          Default: logs/vllm_<jobid>_<port>.log.
#   VLLM_READY_TIMEOUT     Seconds to wait for /v1/models to come up.
#                          Default 1800.
#   VLLM_STALL_TIMEOUT     Seconds with no writes to the vLLM log before we
#                          declare it dead (pairs with PID-check; PID alone
#                          is unreliable on shared HPC nodes due to PID
#                          recycling).  Default 300.
#   VLLM_PROGRESS_EVERY    Seconds between progress lines printed while
#                          waiting for readiness.  Default 30.
#   VLLM_VERSION           Pinned vLLM release to invoke via
#                          `uvx --from vllm==$VLLM_VERSION vllm serve`.
#                          Default 0.19.1 (a known-good wheel for
#                          py3.13/cuda12.9).
#   VLLM_NIGHTLY           1 to ignore VLLM_VERSION and resolve vllm from
#                          the nightly wheel index (https://wheels.vllm.ai/nightly).
#                          Use this only when you specifically need a
#                          main-branch fix; the stable 0.19.1 already
#                          supports Qwen3-8B and Qwen3.5-9B.
#   VLLM_LANGUAGE_MODEL_ONLY  1 to pass `--language-model-only` (skip the
#                          vision tower for VLM-base models we use text-only).
#                          Auto-set for Qwen3.5; default 0 otherwise.
#   VLLM_PREFIX_CACHE      1 to pass `--enable-prefix-caching`. Huge win for
#                          agent loops where each turn's prompt is a prefix
#                          of the next turn's (without it, vLLM re-prefills
#                          the full growing history every turn — quadratic
#                          in n_turns). Default 1; set 0 to opt out.
#   VLLM_ENFORCE_EAGER     1 to pass `--enforce-eager` (skips torch.compile +
#                          cuda-graph capture, so vLLM never shells out to
#                          nvcc).  Auto-enabled when nvcc isn't reachable
#                          after the CUDA env bootstrap; explicitly set to 0
#                          to opt out of the auto behaviour.
#   VLLM_MODULE_LOAD       Space-separated `module load` args.  Default
#                          "gcc/13.4.0 cuda/12.9.1" (Tillicum).  Set "" to
#                          disable module loads entirely.
#   VLLM_CUDA_HOME         CUDA toolkit prefix.  Default
#                          "/gpfs/software/cuda/12.9.1" (Tillicum).  Even when
#                          the modulefile fails to load, we still prepend
#                          $VLLM_CUDA_HOME/bin and lib64 to PATH/LD_LIBRARY_PATH
#                          so `nvcc` and `libcuda.so` are resolvable.  Set ""
#                          to disable.
#   VLLM_AUTO_CAP_MAX_TOKENS  1 to auto-cap the calling shell's $MAX_TOKENS
#                          so vLLM never rejects with
#                          (prompt_tokens + max_tokens) > max_model_len.
#                          Default 1.  Set 0 to opt out entirely (e.g. when
#                          you've manually sized everything via rope scaling).
#   VLLM_MAX_TOKENS_SAFETY_MARGIN  When set, the cap is computed as
#                          (VLLM_MAX_LEN - this).  When unset, the default
#                          behaviour reserves 3/4 of VLLM_MAX_LEN for the
#                          prompt side and uses the remaining 1/4 as the
#                          per-turn generation cap, which is what 16-turn
#                          agent loops need.
#
# Side-effects on _vllm_wait_ready_local: sets HOSTED_VLLM_API_BASE,
# OPENAI_API_KEY, and (if the caller's MODEL didn't already point at
# hosted_vllm/...) overrides MODEL to "hosted_vllm/$VLLM_MODEL".

# This file is sourced; do NOT `set -e` here -- caller has its own.

# --------------------------------------------------------------------------
# No-op stubs first.  The "real" definitions overwrite these when LAUNCH_VLLM=1.
# This way callers can always invoke the functions unconditionally.
# --------------------------------------------------------------------------
_vllm_start_local() { :; }
_vllm_wait_ready_local() { :; }

if [[ "${LAUNCH_VLLM:-0}" != "1" ]]; then
  return 0 2>/dev/null || true
fi

# Module-level state shared across the two functions.
_VLLM_HOST=""
_VLLM_PORT=""
_VLLM_MODEL=""
_VLLM_LOG=""
_VLLM_PID=""
_VLLM_BASE=""

_vllm_cleanup_local() {
  if [[ -n "${_VLLM_PID:-}" ]] && kill -0 "$_VLLM_PID" 2>/dev/null; then
    echo "=== Stopping in-job vLLM (pid=$_VLLM_PID) ==="
    # Negative pid -> whole process group (we used setsid).
    kill -TERM -- "-$_VLLM_PID" 2>/dev/null || kill -TERM "$_VLLM_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$_VLLM_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$_VLLM_PID" 2>/dev/null; then
      kill -KILL -- "-$_VLLM_PID" 2>/dev/null || kill -KILL "$_VLLM_PID" 2>/dev/null || true
    fi
  fi
}

# --------------------------------------------------------------------------
# HPC-cluster CUDA env bootstrap.  vLLM's torch.compile/Inductor pass invokes
# `nvcc` during the KV-cache memory probe at engine startup.  On nodes where
# the CUDA toolkit lives behind a Lmod/EnvironmentModules module (Hyak, NCSA,
# Tillicum, …) `nvcc` is not on PATH by default and vLLM dies with
# `PermissionError: [Errno 13] Permission denied: 'nvcc'`.  We fix it by:
#   1. trying to load the requested gcc/cuda modules (best-effort: if `module`
#      isn't defined, source common Lmod/Modules init files; if those don't
#      exist either, fall through to step 2),
#   2. unconditionally setting CUDA_HOME + prepending its bin/lib64 to PATH/
#      LD_LIBRARY_PATH so `nvcc`/libcuda.so are resolvable even without Lmod,
#   3. setting two vLLM-friendly env vars (CUDA_DEVICE_MAX_CONNECTIONS=1 and
#      VLLM_ALLREDUCE_USE_SYMM_MEM=0) that fix subtle bugs on H200 nodes.
#
# All three steps are overridable via env so other clusters can plug in their
# own module names / paths.  Set VLLM_MODULE_LOAD="" and VLLM_CUDA_HOME="" to
# disable both entirely.
_vllm_setup_cuda_env_local() {
  local module_load="${VLLM_MODULE_LOAD-gcc/13.4.0 cuda/12.9.1}"
  local cuda_home="${VLLM_CUDA_HOME-/gpfs/software/cuda/12.9.1}"

  if [[ -n "$module_load" ]]; then
    # If `module` isn't a function in this shell, try the canonical inits
    # (Lmod first; falls back to env-modules).  Failures here are non-fatal
    # because step 2 (CUDA_HOME) typically suffices on its own.
    if ! command -v module >/dev/null 2>&1; then
      for _init in \
          /etc/profile.d/lmod.sh \
          /usr/share/lmod/lmod/init/bash \
          /etc/profile.d/modules.sh \
          /usr/share/Modules/init/bash; do
        if [[ -r "$_init" ]]; then
          # shellcheck disable=SC1090
          . "$_init" || true
          command -v module >/dev/null 2>&1 && break
        fi
      done
    fi
    if command -v module >/dev/null 2>&1; then
      # shellcheck disable=SC2086
      for _mod in $module_load; do
        if ! module load "$_mod" 2>/dev/null; then
          echo "WARN: module load $_mod failed (continuing; CUDA_HOME fallback may still work)" >&2
        fi
      done
    else
      echo "WARN: \`module\` not found; relying on VLLM_CUDA_HOME=$cuda_home only" >&2
    fi
  fi

  if [[ -n "$cuda_home" ]]; then
    export CUDA_HOME="$cuda_home"
    case ":$PATH:" in
      *":${CUDA_HOME}/bin:"*) ;;
      *) export PATH="${CUDA_HOME}/bin:${PATH}" ;;
    esac
    case ":${LD_LIBRARY_PATH:-}:" in
      *":${CUDA_HOME}/lib64:"*) ;;
      *) export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}" ;;
    esac
  fi

  # vLLM-friendly defaults; users can override either to "" to skip.
  export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
  export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"

  echo "=== CUDA env (post-setup):"
  echo "    CUDA_HOME=${CUDA_HOME:-<unset>}"
  if command -v nvcc >/dev/null 2>&1; then
    echo "    nvcc=$(command -v nvcc) ($(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | head -1))"
  else
    echo "    nvcc=<NOT ON PATH; vllm may need --enforce-eager>"
  fi
  echo "    CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}"
  echo "    VLLM_ALLREDUCE_USE_SYMM_MEM=${VLLM_ALLREDUCE_USE_SYMM_MEM}"
}

_vllm_start_local() {
  if [[ -n "${_VLLM_PID:-}" ]] && kill -0 "$_VLLM_PID" 2>/dev/null; then
    return 0  # idempotent
  fi

  _vllm_setup_cuda_env_local

  _VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
  _VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-8B}"
  local tp="${VLLM_TP:-1}"
  # Qwen3-8B's native max_position_embeddings is 40960, so 40960 is the
  # "free" max -- no rope scaling required.  Bigger values (e.g. 131072 with
  # YaRN) require --rope-scaling and give degraded long-context quality;
  # smaller values waste headroom for agent-style multi-turn loops where
  # most of the budget goes to prompt+tools+history.
  local max_len="${VLLM_MAX_LEN:-40960}"
  local gpu_util="${VLLM_GPU_UTIL:-0.85}"
  local dtype="${VLLM_DTYPE:-bfloat16}"

  # ------------------------------------------------------------------
  # Detect the Qwen3.5 *architecture family*: the set of VLMs sharing the
  # `qwen3_5` HF arch tag (Gated DeltaNet + sparse MoE / hybrid linear/full
  # attention).  Released members so far:
  #   * Qwen3.5-{4B,9B,27B,397B-A17B}  (2026-Mar)
  #   * Qwen3.6-{27B,35B-A3B}          (2026-Apr; same arch tag, same
  #     vLLM serving requirements per the upstream model card recipe).
  # All members need the same three vLLM-level toggles relative to Qwen3:
  #   1. vLLM >= 0.19.0 (per "vllm>=0.19.0; platform_system != 'Darwin'"
  #      constraint shared by Qwen3.5 + Qwen3.6 model cards).  Our 0.19.1
  #      pin already satisfies that, so no nightly needed by default.
  #      Opt into nightly via VLLM_NIGHTLY=1 for bleeding-edge features.
  #   2. --tool-call-parser qwen3_coder (different tool-call format than
  #      Qwen3's hermes format).
  #   3. --language-model-only (skip the vision tower so all GPU memory
  #      goes to KV cache for our text-only agent loop).
  # We detect via a regex matching `qwen3.[5-9]` (case-insensitive on the
  # whole id) so future qwen3_5-arch releases (Qwen3.7, ...) are picked
  # up automatically without script edits.  If a future Qwen drops the
  # qwen3_5 arch tag, set VLLM_TOOL_CALL_PARSER / VLLM_LANGUAGE_MODEL_ONLY
  # explicitly to override.
  # ------------------------------------------------------------------
  local is_qwen3_5=0
  if [[ "${_VLLM_MODEL,,}" =~ qwen3\.[5-9] ]]; then
    is_qwen3_5=1
  fi

  # Tool-call parser: hermes for Qwen2.5/Qwen3, qwen3_coder for the
  # Qwen3.5 arch family (Qwen3.5 / Qwen3.6 / ...) and Qwen3-Coder.
  # `${VAR-default}` (no colon) respects empty-string opt-out.
  local default_tool_call_parser="hermes"
  if [[ "$is_qwen3_5" == "1" ]] || [[ "$_VLLM_MODEL" == *Qwen3-Coder* ]]; then
    default_tool_call_parser="qwen3_coder"
  fi
  local tool_call_parser="${VLLM_TOOL_CALL_PARSER-$default_tool_call_parser}"

  local reasoning_parser
  if [[ -z "${VLLM_REASONING_PARSER+x}" ]]; then
    # qwen3 parser handles Qwen3.x's <think>...</think> format across
    # both Qwen3 and Qwen3.5.
    if [[ "$_VLLM_MODEL" == *Qwen3* ]] || [[ "$_VLLM_MODEL" == *qwen3* ]]; then
      reasoning_parser="qwen3"
    else
      reasoning_parser=""
    fi
  else
    reasoning_parser="$VLLM_REASONING_PARSER"
  fi

  local disable_thinking
  if [[ -z "${VLLM_DISABLE_THINKING+x}" ]]; then
    if [[ "$_VLLM_MODEL" == *Qwen3* ]] || [[ "$_VLLM_MODEL" == *qwen3* ]]; then
      disable_thinking=1
    else
      disable_thinking=0
    fi
  else
    disable_thinking="$VLLM_DISABLE_THINKING"
  fi

  # Skip the vision tower for text-only agent runs (saves a few GB of
  # GPU memory + multimodal preprocessing init).  Qwen3.5 only.
  local language_model_only
  if [[ -z "${VLLM_LANGUAGE_MODEL_ONLY+x}" ]]; then
    language_model_only="$is_qwen3_5"
  else
    language_model_only="$VLLM_LANGUAGE_MODEL_ONLY"
  fi

  # Resolve DP from visible GPU count if not explicitly set.
  local dp ngpu
  if [[ -z "${VLLM_DP:-}" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      ngpu=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c '[0-9]' || true)
    elif command -v nvidia-smi >/dev/null 2>&1; then
      ngpu=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    else
      ngpu=1
    fi
    if [[ -z "$ngpu" || "$ngpu" -lt 1 ]]; then
      ngpu=1
    fi
    dp=$(( ngpu / tp ))
    if [[ "$dp" -lt 1 ]]; then
      dp=1
    fi
  else
    dp="$VLLM_DP"
  fi

  if [[ -n "${VLLM_PORT:-}" ]]; then
    _VLLM_PORT="$VLLM_PORT"
  else
    _VLLM_PORT=$(uv run --quiet python - <<'PYEOF'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PYEOF
)
  fi

  local log_default="logs/vllm_${SLURM_JOB_ID:-local}_${_VLLM_PORT}.log"
  _VLLM_LOG="${VLLM_LOG:-$log_default}"
  mkdir -p "$(dirname "$_VLLM_LOG")"

  # ------------------------------------------------------------------
  # Pin the vLLM version that has a pre-built wheel for our toolchain.
  # Without a pin, uvx tracks the latest release; brand-new
  # releases (we got bit by 0.20.0 on 2026-04-28) often don't ship a wheel
  # for python 3.13 / cuda 12.9 yet, in which case uv transparently *builds
  # vllm from source*, which routinely takes 15+ min and silently aborts.
  #
  # 0.19.1 is the known-good stable for Qwen3-8B AND Qwen3.5-9B (the latter
  # only requires vllm>=0.19.0 per upstream).  Use VLLM_NIGHTLY=1 to opt
  # into the nightly wheel index when you specifically need a bleeding-edge
  # main-branch feature.  Override either via env:
  #   VLLM_VERSION=0.19.1     -- explicit pin (default)
  #   VLLM_NIGHTLY=1          -- use vllm.ai nightly index (opt-in)
  # ------------------------------------------------------------------
  local nightly="${VLLM_NIGHTLY:-0}"

  local vllm_version="${VLLM_VERSION:-0.19.1}"
  local cmd
  if [[ "$nightly" == "1" ]]; then
    # Use vLLM's nightly wheel index.  Don't pin a version (let uv pick the
    # latest available).  Also force --prerelease=allow so uv accepts
    # `0.21.0.dev123+...` style nightly versions.
    cmd=(uvx --extra-index-url "https://wheels.vllm.ai/nightly"
         --prerelease=allow
         --from "vllm" vllm serve "$_VLLM_MODEL")
    vllm_version="<nightly>"
  else
    cmd=(uvx --from "vllm==${vllm_version}" vllm serve "$_VLLM_MODEL")
  fi
  cmd+=(--host "$_VLLM_HOST"
        --port "$_VLLM_PORT"
        --tensor-parallel-size "$tp"
        --max-model-len "$max_len"
        --gpu-memory-utilization "$gpu_util"
        --dtype "$dtype"
        --served-model-name "$_VLLM_MODEL")
  if [[ "$dp" -gt 1 ]]; then
    cmd+=(--data-parallel-size "$dp")
  fi
  if [[ -n "$tool_call_parser" ]]; then
    cmd+=(--enable-auto-tool-choice --tool-call-parser "$tool_call_parser")
  fi
  if [[ -n "$reasoning_parser" ]]; then
    cmd+=(--reasoning-parser "$reasoning_parser")
  fi
  # Skip the vision encoder for VLM-base models we're using as text-only
  # (Qwen3.5).  Frees a few GB of GPU memory + skips multimodal preprocessor
  # init.  Recognized by vLLM nightly; older releases don't have this flag
  # but won't hit this branch since they don't run Qwen3.5 anyway.
  if [[ "$language_model_only" == "1" ]]; then
    cmd+=(--language-model-only)
  fi
  # Prefix caching — huge win for agent loops where each turn's prompt is a
  # prefix of the next turn's prompt. Without it, vLLM re-prefills the full
  # growing history every turn (quadratic in n_turns). With it, only the
  # NEW tokens are processed. Real-world impact for our 16-action / 60-action
  # agent loops is 2-3× total throughput on the SFT solve runs. Default ON;
  # set VLLM_PREFIX_CACHE=0 to opt out (e.g. if a corner case ever breaks).
  if [[ "${VLLM_PREFIX_CACHE:-1}" == "1" ]]; then
    cmd+=(--enable-prefix-caching)
  fi
  # Optional escape hatch: skip torch.compile + CUDA-graph capture entirely so
  # vLLM never invokes nvcc at engine init.  Costs ~10-30% throughput but is
  # bulletproof on nodes where the CUDA toolkit isn't reachable.  Auto-enable
  # only when nvcc still isn't on PATH after the CUDA env bootstrap above.
  local enforce_eager
  if [[ -z "${VLLM_ENFORCE_EAGER+x}" ]]; then
    if command -v nvcc >/dev/null 2>&1; then
      enforce_eager=0
    else
      enforce_eager=1
      echo "INFO: nvcc still missing after CUDA env bootstrap -> auto-enabling --enforce-eager." >&2
    fi
  else
    enforce_eager="$VLLM_ENFORCE_EAGER"
  fi
  if [[ "$enforce_eager" == "1" ]]; then
    cmd+=(--enforce-eager)
  fi
  # NOTE: We deliberately do NOT pass `--chat-template-kwargs` on the CLI here.
  # That flag was only added in newer vLLM (>=0.7.x); older `uvx vllm` builds
  # error out with "unrecognized arguments: --chat-template-kwargs ...".
  # Instead, when $disable_thinking == 1 we propagate the same intent to the
  # *request body* via $LITELLM_EXTRA_BODY_JSON (see _vllm_wait_ready_local
  # below + rl_data/__init__.py), which is universally accepted by every vLLM
  # server that supports Qwen3's `enable_thinking` chat-template kwarg.
  if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    local extra=( $VLLM_EXTRA_ARGS )
    cmd+=("${extra[@]}")
  fi

  echo "=== Starting in-job vLLM ==="
  echo "  model         : $_VLLM_MODEL"
  echo "  bind          : $_VLLM_HOST:$_VLLM_PORT"
  echo "  tp x dp       : $tp x $dp"
  echo "  max_model_len : $max_len"
  echo "  dtype         : $dtype"
  echo "  tool parser   : ${tool_call_parser:-<disabled>}"
  echo "  reason parser : ${reasoning_parser:-<disabled>}"
  echo "  enforce_eager : $enforce_eager"
  echo "  no_thinking   : $disable_thinking"
  echo "  qwen3.5+ arch : is_qwen3_5=$is_qwen3_5  language_only=$language_model_only  nightly=$nightly"
  echo "  vllm version  : ${vllm_version:-<latest>}"
  echo "  log           : $_VLLM_LOG"
  echo "  cmd           : ${cmd[*]}"
  echo

  # Start vLLM in its own session so we can SIGTERM the whole tree on exit.
  setsid "${cmd[@]}" </dev/null >"$_VLLM_LOG" 2>&1 &
  _VLLM_PID=$!
  _VLLM_BASE="http://${_VLLM_HOST}:${_VLLM_PORT}/v1"

  trap _vllm_cleanup_local EXIT INT TERM
  echo "=== vLLM background pid=$_VLLM_PID; will wait for readiness later ==="
  echo
}

_vllm_wait_ready_local() {
  if [[ -z "${_VLLM_PID:-}" ]]; then
    echo "ERROR: _vllm_wait_ready_local called before _vllm_start_local." >&2
    return 1
  fi

  local timeout="${VLLM_READY_TIMEOUT:-1800}"
  local stall_timeout="${VLLM_STALL_TIMEOUT:-300}"   # seconds with NO log writes -> declare dead
  local progress_every="${VLLM_PROGRESS_EVERY:-30}"  # seconds between status prints
  local deadline=$(( $(date +%s) + timeout ))
  local last_progress=$(date +%s)
  local last_size=0

  echo "=== Waiting for vLLM to become ready"
  echo "    overall timeout    : ${timeout}s"
  echo "    stall timeout      : ${stall_timeout}s (declare dead if log doesn't grow for this long)"
  echo "    progress reports   : every ${progress_every}s"
  echo "    log file           : $_VLLM_LOG"
  while true; do
    # 1. Liveness via PID -- best-effort; PIDs can be recycled on busy hosts
    #    so we ALSO use the log-progress check below as the authoritative signal.
    local pid_alive=0
    if kill -0 "$_VLLM_PID" 2>/dev/null; then
      pid_alive=1
    fi

    # 2. Readiness probe.
    if curl -sS --max-time 3 -o /dev/null -w "%{http_code}" \
         "${_VLLM_BASE}/models" 2>/dev/null | grep -q '^200$'; then
      echo "=== vLLM ready at ${_VLLM_BASE} ==="
      break
    fi

    # 3. Stall detector: track log file mtime + size.  If the log hasn't been
    #    written to for $stall_timeout seconds AND the PID-check says dead,
    #    declare failure.  This catches uvx silently aborting mid-build (a
    #    PID recycle on a busy node fools a pure kill -0 check).
    local now=$(date +%s)
    local cur_size=0
    local log_mtime=0
    if [[ -f "$_VLLM_LOG" ]]; then
      cur_size=$(stat -c '%s' "$_VLLM_LOG" 2>/dev/null || echo 0)
      log_mtime=$(stat -c '%Y' "$_VLLM_LOG" 2>/dev/null || echo 0)
    fi
    local log_idle=$(( now - log_mtime ))

    if [[ "$pid_alive" == "0" && "$log_idle" -ge "$stall_timeout" ]]; then
      echo "ERROR: vLLM looks dead (pid $_VLLM_PID gone AND log idle ${log_idle}s >= ${stall_timeout}s)." >&2
      echo "       Tail of $_VLLM_LOG:" >&2
      tail -n 80 "$_VLLM_LOG" >&2 || true
      return 1
    fi
    # Pure log-stall fallback (PID may be recycled to an unrelated process):
    if [[ "$log_idle" -ge "$(( stall_timeout * 2 ))" ]]; then
      echo "ERROR: vLLM log idle ${log_idle}s (>= ${stall_timeout}s x 2) -- assuming silent crash." >&2
      echo "       Tail of $_VLLM_LOG:" >&2
      tail -n 80 "$_VLLM_LOG" >&2 || true
      return 1
    fi

    # 4. Periodic progress so the user sees motion (or lack thereof).
    if [[ $(( now - last_progress )) -ge "$progress_every" ]]; then
      local last_line
      last_line=$(tail -n 1 "$_VLLM_LOG" 2>/dev/null | tr -d '\r' | cut -c1-120)
      printf '  [%(%H:%M:%S)T] elapsed=%ss  log_size=%s  log_idle=%ss  pid_alive=%s  last="%s"\n' \
        -1 "$(( now - (deadline - timeout) ))" "$cur_size" "$log_idle" "$pid_alive" "${last_line:-<empty>}"
      last_progress=$now
    fi

    # 5. Overall timeout.
    if [[ "$now" -ge "$deadline" ]]; then
      echo "ERROR: vLLM did not become ready within ${timeout}s. Tail of $_VLLM_LOG:" >&2
      tail -n 80 "$_VLLM_LOG" >&2 || true
      return 1
    fi
    sleep 5
  done

  export HOSTED_VLLM_API_BASE="$_VLLM_BASE"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
  if [[ "${MODEL:-}" != hosted_vllm/* ]]; then
    export MODEL="hosted_vllm/${_VLLM_MODEL}"
  fi

  # ----------------------------------------------------------------------
  # Auto-cap MAX_TOKENS so vLLM never rejects with prompt_tokens+max_tokens
  # > max_model_len.  Critical for AGENT-STYLE LOOPS: a 16-turn bash-tool
  # agent piles up tool schemas + system prompt + N-turn history on the
  # prompt side, easily 20k+ tokens by mid-loop.  The naive cap
  # `max_tokens = max_len - 4096` therefore breaks after turn 1-2 with
  # ContextWindowExceededError.  Default formula instead reserves 3/4 of
  # the model's window for the prompt and gives the remaining 1/4 to the
  # per-turn generation cap; for Qwen3-8B (40960) that's 10240 max_tokens
  # output / 30720 prompt, plenty for the harness's max_actions=16 budget.
  #
  # Knobs:
  #   VLLM_AUTO_CAP_MAX_TOKENS=0      -- skip the cap entirely
  #   VLLM_MAX_TOKENS_SAFETY_MARGIN=N -- reserve a fixed N tokens for input
  #                                       (cap = max_len - N) instead of
  #                                       the default 3/4-of-context rule.
  # ----------------------------------------------------------------------
  if [[ "${VLLM_AUTO_CAP_MAX_TOKENS:-1}" == "1" && -n "${MAX_TOKENS:-}" ]]; then
    local vllm_max_len="${VLLM_MAX_LEN:-40960}"
    local cap reason
    if [[ -n "${VLLM_MAX_TOKENS_SAFETY_MARGIN:-}" ]]; then
      cap=$(( vllm_max_len - VLLM_MAX_TOKENS_SAFETY_MARGIN ))
      reason="VLLM_MAX_LEN=${vllm_max_len}, fixed safety margin=${VLLM_MAX_TOKENS_SAFETY_MARGIN}"
    else
      # Default: reserve 3/4 of context for prompt+tools+agent history.
      cap=$(( vllm_max_len / 4 ))
      reason="VLLM_MAX_LEN=${vllm_max_len}, reserving 3/4 for prompt+history"
    fi
    if [[ "$cap" -lt 1024 ]]; then
      cap=1024  # never cap below something usable
    fi
    if [[ "$MAX_TOKENS" -gt "$cap" ]]; then
      echo "INFO: capping MAX_TOKENS from ${MAX_TOKENS} to ${cap} (${reason})"
      MAX_TOKENS="$cap"
    fi
  fi

  # Propagate enable_thinking=false (when requested) to the litellm wrapper
  # via the request body, since older vLLM builds don't have the equivalent
  # CLI flag.  Read by rl_data.__init__.chat_completion_batch_with_tools.
  local disable_thinking
  if [[ -z "${VLLM_DISABLE_THINKING+x}" ]]; then
    if [[ "$_VLLM_MODEL" == *Qwen3* ]] || [[ "$_VLLM_MODEL" == *qwen3* ]]; then
      disable_thinking=1
    else
      disable_thinking=0
    fi
  else
    disable_thinking="$VLLM_DISABLE_THINKING"
  fi
  if [[ "$disable_thinking" == "1" ]]; then
    export LITELLM_EXTRA_BODY_JSON='{"chat_template_kwargs": {"enable_thinking": false}}'
  fi

  echo "=== Solver will use:"
  echo "    MODEL=$MODEL"
  echo "    HOSTED_VLLM_API_BASE=$HOSTED_VLLM_API_BASE"
  if [[ -n "${LITELLM_EXTRA_BODY_JSON:-}" ]]; then
    echo "    LITELLM_EXTRA_BODY_JSON=$LITELLM_EXTRA_BODY_JSON"
  fi
  echo
}
