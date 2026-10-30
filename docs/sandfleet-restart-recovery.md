# Sandfleet controller restart recovery

Set `SANDFLEET_REGISTRY` to the controller's `--registry` JSON file on shared
storage visible at the same absolute path to every Ray actor and worker. Keep
`SANDFLEET_URL` as a bootstrap address and preserve role tokens and controller
state across restarts. Controller requests reread the registry on each attempt;
worker data-plane commands stay on their original agent URL and are not replayed.

Deploy with Sandfleet 0.7.5's worker discovery changes between runs. Existing
worker processes also need the updated code and registry path; changing only the
controller does not retrofit recovery.
Renewal occurs at most 30 seconds apart (sooner for short TTLs), with transient
retries bounded by 300 seconds and half the lease TTL. Individual renewal calls
are capped at 10 seconds. Permanent errors and exhausted retries surface as
infrastructure failures. Registry syntax/authentication errors are not hidden.
Persisted lease expiry remains authoritative; outages longer than these bounds
can still lose sandboxes. This does not provide multi-controller failover.
