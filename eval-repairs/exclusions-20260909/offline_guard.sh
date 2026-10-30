# Preserve verifier output and status without inferring its cause from text.
# Dependency readiness must be validated before model execution, separately.
run_offline_verifier() {
    local output status
    output=$(mktemp) || {
        echo 'SETUP-FAILED: cannot create verifier output file' >&2
        exit 90
    }
    "$@" >"$output" 2>&1
    status=$?
    cat "$output"
    rm -f "$output"
    return "$status"
}
