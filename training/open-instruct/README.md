# open-instruct (fork)

This is a fork of [allenai/open-instruct](https://github.com/allenai/open-instruct).

It contains fixes on top of upstream for:

- **Qwen 3.5** support (fixed hybrid CP-SP training for SFT and RL)
- **DPPO Support** (new RL loss)
- **Terminal agent training** (podman-based sandboxes for training)

The training scripts for this fork live under [`training/open-instruct/scripts/tmax`](scripts/tmax). Please refer to the [README](scripts/tmax/README.md) for more details on how to use them. Note that we made this code and infra for training at Ai2, so you may need to modify some things to run it on your own infrastructure. For example, swapping to apptainer for sandboxing might be required, which we do not really officially support in this code. I recommend starting with the 1 GPU RL debug script (`qwen35_2b_1gpu.sh`), getting that working, and then scaling up to the full-size scripts.

### Slurm sandbox pools with Sandfleet

For large sandbox pools that should not consume the training allocation's CPU
and memory, `backend: "sandfleet"` leases a sandbox from a separate Sandfleet
service. The service owns the Slurm worker allocations; TMAX only needs its
private URL and client token:

```bash
export SANDFLEET_URL=http://sandfleet-controller:8765
export SANDFLEET_CLIENT_TOKEN=...
```

```json
{
  "backend": "sandfleet",
  "mem_limit": "4g",
  "sandfleet_acquire_timeout": 900,
  "image": "/shared/images/tmax.sif"
}
```

TMAX requests two CPUs and uses the existing `mem_limit` as the Sandfleet RAM
request. Sandfleet validates that shape against controller policy and routes
matching requests into one reusable homogeneous pool. TMAX never sends raw
Slurm flags. A deployment can set `sandfleet_pool` to use a pre-created named
pool instead; in that mode the pool profile defines resources and `mem_limit`
is ignored. Sandfleet returns a scoped lease whose Apptainer instance can be
restarted without submitting another worker job. GPU selection remains a
Sandfleet service feature until TMAX has a real GPU-sandbox use case.

If an elastic pool has no immediately ready slot, reset waits for Sandfleet to
start capacity up to `sandfleet_acquire_timeout`. If Slurm preempts the worker
hosting an active sandbox, the affected rollout ends with `sandbox_lost` and
`infrastructure_failure` metadata; the next reset acquires replacement
capacity rather than replaying a potentially non-idempotent command.

For general documentation, usage, and the upstream codebase, refer to the [main open-instruct repository](https://github.com/allenai/open-instruct). I also recommend checking it for the flags and features.

### Requirements

- `uv` for dependency management (deps pinned in the repo-root `pyproject.toml` / `uv.lock`).
- A Dockerhub login and personal access token (PAT). In particular, you probably need a business account to pull images from Dockerhub at large scale.
