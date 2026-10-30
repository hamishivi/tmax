# Vanillux evaluation behavior

Vanillux evaluation and training share the think-before-acting prompt,
persistent cwd and exports, first-tool-call dispatch, and format-error recovery.
Per-exec fakeroot connection keys are excluded from saved shell state.

Nonzero shell initialization fails explicitly. Backend command deadlines retain
partial output and timeout metadata. Evaluation permits a subsequent agent turn;
Harbor owns the overall task deadline and final verification. A timed-out
command cannot submit the task. Ordinary exit code 124 is distinguishable from
an explicit backend deadline.

Sandfleet integrations must preserve the backend's explicit `timed_out` field.
The patches in `eval-repairs/sandfleet-0.7.9-harbor-timeouts.patch` and
`eval-repairs/sandfleet-0.7.9-command-argv.patch` support that adapter version and
pass quote-heavy commands as separate arguments rather than nested shell text.
Use a matched client, controller, worker, and Harbor adapter version.

Daytona agent commands use file-based capture so background services do not
hold SDK output pipes open. Command-group cancellation preserves earlier
services and partial output. The capture identifier is protected from
command-text process matching. This works through stock Harbor's
`environment.exec()` interface; setup and verifier execution are unchanged.

Training's reward filtering, actor replacement, task setup, and reference-policy
checkpointing remain trainer functions. Benchmark resources, initialization,
working directories, and original graders remain controlled by the task and
Harbor. Different image or dependency variants should retain separate manifests.

The regression tests in `tests/` use fake transports and local Bash processes;
they require no model or live sandbox service. Process-group tests require Linux.
