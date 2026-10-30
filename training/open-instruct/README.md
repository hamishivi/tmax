# open-instruct (fork)

This is a fork of [allenai/open-instruct](https://github.com/allenai/open-instruct).

It contains fixes on top of upstream for:

- **Qwen 3.5** support (fixed hybrid CP-SP training for SFT and RL)
- **DPPO support** with capped importance ratios and distributed batch padding
- **Terminal agent training** with valid-sibling rewards, persistent shells, independent grading deadlines, and Docker/Podman, Apptainer, or Sandfleet sandboxes
- **Pretokenized SFT input** via `--pretokenized_jsonl`

The training recipes live under [`scripts/tmax`](scripts/tmax). See their
[README](scripts/tmax/README.md) for runtime configuration, dataset overrides,
Sandfleet setup, and pretokenized SFT input. Start with the single-GPU RL debug
recipe (`qwen35_2b_1gpu.sh`) before scaling to the full GPU layouts.

For general documentation, usage, and the upstream codebase, refer to the [main open-instruct repository](https://github.com/allenai/open-instruct). I also recommend checking it for the flags and features.

### Requirements

- `uv` for dependency management (deps pinned in the repo-root `pyproject.toml` / `uv.lock`).
- A Dockerhub login and personal access token (PAT). In particular, you probably need a business account to pull images from Dockerhub at large scale.
