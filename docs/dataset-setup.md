# Optional dataset setup

The Vanillux training environment accepts an optional string `setup_command` in
per-row `env_config` (passed to `reset`). Missing, null, empty, or whitespace-only
commands do nothing. No image name or dataset name triggers implicit setup.

Setup runs after instruction/seed loading and shell initialization, before the
first action. Seeds are available under `/workspace`; verifier tests stay hidden
until submission. Setup uses the normal command timeout (`timeout`, default 120
seconds). A nonzero exit or timeout fails reset and follows its existing retry
policy; it is not a task reward. Failures report task, exit status, elapsed time,
and bounded diagnostic output.

Use Bash commands with explicit error propagation (`&&`, or `set -e`). Exported
variables and `cd` persist to agent actions. The verifier starts with a snapshot
of the successful setup environment and cwd, independent of later agent shell
changes. Do not use `exit`, `exec`, or background initialization on a successful
setup path: the wrapper must finish saving its shell state. Each reset reruns the
current row's command in the fresh sandbox. These commands execute trusted dataset
code, just like the dataset's verifier; there is no task-id interpolation by the
harness.

For example, a dataset can ship a setup script under `environment/seeds` and
set `env_config["setup_command"] = "source /workspace/setup.sh"`. Sourcing the
script preserves its exports and working directory. Dataset scripts should use
`set -e` or explicit checked commands, and quote task-derived arguments.

## SWE-smith integration requirements

Checkout alone is not a complete SWE-smith adapter. At upstream source revision
`9b74ac08118a85c39c356802f7961893af73e07f`, the official evaluation harness:

- Fetches task refs before checkout. Resolve refs to immutable commits when
  preparing campaign artifacts; avoid a network fetch for every rollout.
- Evaluates from the bug commit beneath the test-removal commit (`HEAD~1` in
  upstream's documented two-commit layout). Preserve the task branch state for
  the agent, then restore the authoritative test files at verification time.
  Do not blindly move the agent to the parent commit or undo agent source edits.
- Reverts prediction changes to test files before grading. An in-place training
  verifier must similarly restore authoritative tests without reverting the fix.
- Uses the repository profile's environment activation, test command, and log
  parser. Go tasks use Go tests, not pytest; some Python profiles override the
  default command as well.
- Checks both `FAIL_TO_PASS` and `PASS_TO_PASS` by default. Missing expected test
  results count as failures; exit zero or an empty test selection is insufficient.

Use the pinned upstream profile/harness to prepare setup and verifier artifacts,
and record any deliberate reward-policy difference. The generic setup hook does
not implement these dataset semantics. Test artifacts remain deferred until
submission. Validate no-op, known repair, regression, and modified/missing-test
cases before launching the converted dataset.

References: [upstream execution](https://github.com/SWE-bench/SWE-smith/blob/9b74ac08118a85c39c356802f7961893af73e07f/swesmith/harness/utils.py),
[upstream grading](https://github.com/SWE-bench/SWE-smith/blob/9b74ac08118a85c39c356802f7961893af73e07f/swesmith/harness/grading.py).
