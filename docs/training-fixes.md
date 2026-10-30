# Portable training and sandbox behavior

The trainer supports capped DPPO, masked padding for distributed batches,
explicit reward validity, and Sandfleet sandboxes. Service addresses, access
tokens, image pools, and task-data locations are supplied by the operator.

## DPPO and distributed batches

With `loss_fn=dppo`, `dppo_ratio_cap` defaults to 10. The policy-gradient
coefficient uses a detached capped importance ratio; zero disables the cap.
Log ratios are bounded to `[-20, 20]` before exponentiation. Both dense and
tiled loss paths use the same rule and report policy-drift statistics.

Packed batches retain every real sequence. Additional packs have masked
supervision and sentinel rollout identities so all data-parallel workers enter
the same collectives without adding training signal. ZeRO-3 and sequence
parallelism use the appropriate gradient-reduction divisor, including empty
response shards. The tiled loss supports entropy monitoring.

Reference-policy updates operate on matching local ZeRO-3 shards. Checkpoint
save/load gathers complete state collectively, validates keys and shapes before
loading, synchronizes buffers, and propagates errors consistently across ranks.
A load that fails after mutation begins is fatal rather than silently falling
back to a partially overwritten reference policy.

## Reward validity and collection

Infrastructure failures and missing, nonfinite, or out-of-range verifier rewards
are unscored outcomes. They do not become ordinary zero-reward examples.
Advantages use the valid siblings in each prompt group, and invalid siblings are
removed before packing. Groups with too few valid completions are resampled.
Timeout outcomes retain a verifier reward when one was actually produced.

Population metrics count sampled valid completions before zero-variance group
selection, including constant-reward groups. Post-filter population metrics
apply the configured overlong filter. Counts accompany means; an empty
population has no fabricated mean.

`async_steps=0` selects learner-driven synchronous collection. Positive values
retain asynchronous prefetching. Weight synchronization is tracked explicitly,
and background generation failures reach the learner instead of leaving it
waiting indefinitely. Trace saving snapshots result lists before asynchronous
filtering can replace them.

`vllm_seed_per_turn` defaults to false. Enabling it assigns distinct deterministic
seeds to tool-generation turns. Record this option when reproducing a run.

## Sandfleet configuration

Select `backend="sandfleet"` in the sandbox environment configuration. Supply
`SANDFLEET_URL` and `SANDFLEET_CLIENT_TOKEN` through the environment. The client
leases sandboxes and handles bounded transient retries, lease renewal, command
execution, file transfer, and explicit worker-loss errors. Optional
`SANDFLEET_REGISTRY` supports controller address recovery; see
[controller recovery](sandfleet-restart-recovery.md).

Set `SWERL_APPTAINER_SIF_DIR` to a local image pool shared with the workers.
Container references resolve to exact names or matching hash names in that
pool. Missing local images fail explicitly. Remote fallback requires
`SWERL_ALLOW_REMOTE_IMAGE_FALLBACK=1`.

Environment pools replace discarded actors and surface replacement failures.
Timed-out calls are drained or quarantined before their actor can be reused.
The optional `SWERL_ENV_CLOSE_ON_RELEASE` setting releases idle sandbox resources.

## Execution and deadlines

The Vanillux shell preserves cwd and user exports between calls, without
persisting the per-exec `FAKEROOTKEY`. Optional dataset setup is checked and
snapshotted for verification; see [dataset setup](dataset-setup.md).
Only the first tool call is dispatched, and malformed calls receive feedback
within the ordinary turn budget. The prompt asks the policy to think before
acting without requiring a heading.

`SWERL_ROLLOUT_TIMEOUT_S` defaults to 600 seconds; nonpositive values disable the
whole-rollout limit. Its clock starts before sandbox acquisition and reset.
Generation and ordinary tools use the remaining rollout budget. Acquisition
itself is not interrupted by this deadline; an expired budget is checked after
reset. Command cancellation targets the active command group and preserves
partial output.

Sandbox `pwd` may be configured directly or through a JSON `pwd_map` from task
IDs to absolute paths. A supplied map must cover every selected task.
An optional JSON `verifier_timeout_map` assigns positive per-task budgets;
`max_verifier_timeout_s` can impose an explicit ceiling. Test assets are uploaded
to the location expected by the selected task's verifier.

Recognized submissions receive the selected verifier budget plus bounded
transport/cleanup overhead, separately from ordinary tool and remaining agent
deadlines. Logs distinguish command timeout, rollout timeout, verifier timeout,
and grading abandoned by its caller. A submission marker generated only in
command output cannot be recognized in advance; that fallback is logged
separately if grading exceeds the ordinary tool deadline.

## Pretokenized SFT input

`finetune.py --pretokenized_jsonl /path/to/trajectories.jsonl` loads tokenized
records with equal-length `input_ids` and `labels`. It adds an all-ones attention
mask and filters examples beyond `max_seq_length`. This preserves tool and
special-token boundaries without retokenizing saved trajectories.

## Evaluation

[Evaluation dependency repairs](../eval-repairs/README.md) provide offline image
and bootstrap recipes without changing task assertions or model-visible task
content. [Vanillux evaluation](vanillux-evaluation-parity.md) describes shell,
timeout, and Daytona output-capture behavior.
