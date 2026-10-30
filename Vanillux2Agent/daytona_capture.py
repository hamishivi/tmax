"""Capture agent commands through stock Harbor, without inherited output pipes."""

import asyncio
import base64
import shlex
import uuid


async def execute_daytona_command(environment, command: str, timeout: int):
    """Return stdout, stderr, exit code and explicit deadline evidence.

    The command budget is enforced inside our wrapper. Harbor gets extra time
    for cleanup and output transfer; the host guard bounds transport stalls.
    No Harbor internals or Daytona SDK methods are used.
    """
    if timeout <= 0:
        raise ValueError("Daytona command timeout must be positive")
    prefix = "/tmp/.vanillux2-command-" + uuid.uuid4().hex
    done = shlex.quote(prefix + ".done")
    group = shlex.quote(prefix + ".pgid")
    tag = "__vanillux2_completion_" + uuid.uuid4().hex + "__"
    # Keep model text out of capture/timeout ancestors' command lines. Broad
    # pkill -f patterns may kill the inner user shell, but must not kill the
    # supervisor that returns its exit status and collected output.
    encoded = base64.b64encode(command.encode("utf-8")).decode("ascii")
    # The sentinel preserves trailing newlines through command substitution.
    user_shell = (
        "payload=$(printf %s "
        + shlex.quote(encoded)
        + " | base64 -d && printf .) || exit 125\n"
        + 'bash -c "${payload%.}"'
    )
    supervisor = (
        "read -r pid comm state ppid pgid rest < /proc/$$/stat\n"
        + f'printf "%s" "$pgid" > {group}\n'
        + user_shell
        + "\n"
        + "rc=$?\n"
        + f"printf done > {done}\n"
        + 'exit "$rc"'
    )
    capture = (
        "d=$(mktemp -d) || exit 125\n"
        + f"timeout --kill-after=5s {timeout} bash -c "
        + shlex.quote(supervisor)
        + ' >"$d/out" 2>"$d/err"\n'
        + "rc=$?\n"
        + f'if [ ! -f {done} ] && {{ [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; }}; then\n'
        + f"  pgid=$(cat {group}) || exit 125\n"
        + '  case "$pgid" in ""|*[!0-9]*) exit 125;; esac\n'
        + '  kill -KILL "-$pgid" 2>/dev/null || { kill -0 "-$pgid" 2>/dev/null && exit 125; }\n'
        + "fi\n"
        + 'head -c "$(wc -c < "$d/out")" "$d/out"\n'
        + 'head -c "$(wc -c < "$d/err")" "$d/err" >&2\n'
        + 'rm -rf "$d"\n'
        + f"if [ -f {done} ]; then printf '%s' {shlex.quote(tag + '1')} >&2; "
        + f"else printf '%s' {shlex.quote(tag + '0')} >&2; fi\n"
        + f"rm -f {done} {group}\n"
        + 'exit "$rc"'
    )
    result = await asyncio.wait_for(environment.exec(command=capture, timeout_sec=timeout + 15), timeout=timeout + 30)
    # Daytona adds a newline to the final log record. The tag separates framing
    # from user stderr, including intentionally absent or repeated newlines.
    stderr = (result.stderr or "").rstrip("\r\n")
    if not stderr.endswith((tag + "0", tag + "1")):
        raise RuntimeError("Missing Daytona command completion evidence")
    completed = stderr.endswith(tag + "1")
    stderr = stderr[: -(len(tag) + 1)]
    timed_out = not completed and result.return_code in (124, 137)
    return result.stdout, stderr, 124 if timed_out else result.return_code, timed_out
