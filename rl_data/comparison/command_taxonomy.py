"""Classify bash commands into a small, stable taxonomy.

Used by `compare_datasets.py` to characterise the *kinds* of actions agents
take when solving tasks in each dataset (rather than treating every shell
invocation as opaque).

The taxonomy is deliberately coarse; the goal is to produce interpretable
stacked-bar and coverage plots that separate "mostly file manipulation"
datasets from ones that require code, services, databases, networking, etc.

Each command is split on `;`, `&&`, `||`, `|`, and newlines so multi-step
one-liners contribute multiple tags.  Each sub-command is tagged with zero
or more categories; an unclassified sub-command contributes the `other`
tag so we can monitor taxonomy coverage.

Categories (in the order plots should display them):

    shell_nav       cd / ls / pwd / which / find
    file_read       cat / less / more / head / tail
    file_write      echo>...  printf>...  tee  truncate
    file_manip      mv / cp / rm / mkdir / touch / ln
    text_proc       sed / awk / grep / cut / sort / uniq / tr / wc
    archive         tar / zip / unzip / gzip / gunzip / bzip2
    perm            chmod / chown / chgrp / umask
    code_write      heredoc (<<'EOF') to .py/.js/.c/... or `cat > foo.py`
    code_run        python / python3 / node / gcc / g++ / clang / make / cargo / go / javac / rustc / ruby / perl / bash-script
    pkg_install     apt / apt-get / pip / pip3 / npm / yarn / cargo install / go install / gem install
    service         systemctl / service / nginx / apache / redis-server / postgres / mysqld / supervisord
    db              sqlite3 / psql / mysql / mongo / redis-cli
    net             curl / wget / ssh / scp / nc / netcat / ping / dig / host
    env_var         export / env / source
    verify          pytest / python -m pytest / sha256sum / diff / test -/ [ -f / cmp
    other           (nothing matched)
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Set

# Ordered for consistent plot legends. `other` is always last.
CATEGORIES: List[str] = [
    "shell_nav",
    "file_read",
    "file_write",
    "file_manip",
    "text_proc",
    "archive",
    "perm",
    "code_write",
    "code_run",
    "pkg_install",
    "service",
    "db",
    "net",
    "env_var",
    "verify",
    "other",
]

# Map of category -> set of bash "head tokens" (argv[0] basename, or equivalent).
_HEAD_TOKENS: dict[str, Set[str]] = {
    "shell_nav": {"cd", "ls", "pwd", "which", "whereis", "find", "tree", "stat", "file", "type"},
    "file_read": {"cat", "less", "more", "head", "tail", "bat"},
    "file_manip": {"mv", "cp", "rm", "rmdir", "mkdir", "touch", "ln", "install"},
    "text_proc": {"sed", "awk", "grep", "egrep", "fgrep", "rg", "cut", "sort", "uniq",
                  "tr", "wc", "paste", "join", "comm", "column", "tee", "xargs", "jq", "yq"},
    "archive": {"tar", "zip", "unzip", "gzip", "gunzip", "bzip2", "bunzip2", "xz", "7z"},
    "perm": {"chmod", "chown", "chgrp", "umask", "setfacl", "getfacl"},
    "code_run": {
        "python", "python3", "py", "ipython", "pypy", "pypy3",
        "node", "nodejs", "npx", "deno", "bun",
        "gcc", "g++", "clang", "clang++", "cc", "c++", "make", "cmake",
        "cargo", "rustc",
        "go", "go-run",
        "javac", "java", "scalac", "scala", "kotlin", "kotlinc",
        "ruby", "irb", "perl", "php", "lua", "luajit",
        "rscript", "octave", "dotnet",
        "sh", "bash", "zsh", "fish", "ash",  # running shell scripts
    },
    "pkg_install": set(),  # handled via substring check below
    "service": {"systemctl", "service", "nginx", "apache2", "httpd",
                "redis-server", "postgres", "postgresql", "mysqld", "mongod",
                "supervisord", "supervisorctl", "docker", "dockerd"},
    "db": {"sqlite3", "psql", "mysql", "mongo", "mongosh", "redis-cli",
           "cqlsh", "clickhouse-client"},
    "net": {"curl", "wget", "ssh", "scp", "rsync", "nc", "netcat", "ncat",
            "ping", "dig", "host", "nslookup", "telnet", "ftp", "sftp",
            "ip", "ifconfig", "iptables"},
    "env_var": {"export", "env", "source", "unset"},
    "verify": {"pytest", "sha256sum", "md5sum", "sha1sum", "diff", "cmp",
               "assert", "expect"},
}

# Patterns that match anywhere in the command string (regex).
_SUBSTRING_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Package managers (covers "apt-get install", "pip3 install", etc.)
    ("pkg_install", re.compile(r"\b(apt(-get)?|yum|dnf|apk|brew|zypper|pacman)\s+(-\S+\s+)*install\b")),
    ("pkg_install", re.compile(r"\bpip3?\s+install\b")),
    ("pkg_install", re.compile(r"\bnpm\s+(install|i|ci|add)\b")),
    ("pkg_install", re.compile(r"\byarn\s+(add|install)\b")),
    ("pkg_install", re.compile(r"\bpnpm\s+(add|install)\b")),
    ("pkg_install", re.compile(r"\bcargo\s+install\b")),
    ("pkg_install", re.compile(r"\bgo\s+install\b")),
    ("pkg_install", re.compile(r"\bgem\s+install\b")),
    ("pkg_install", re.compile(r"\buv\s+(add|pip\s+install)\b")),
    ("pkg_install", re.compile(r"\bpoetry\s+add\b")),

    # File-write via redirection or tee.  Covers:
    #   echo "..." > foo          printf ... >> foo
    #   cat > foo.py              cat <<'EOF' > foo
    #   python -c '...' > foo
    ("file_write", re.compile(r">>?\s*['\"]?[\w./\-]+['\"]?")),
    ("file_write", re.compile(r"\btee\s+")),

    # Heredoc -> treat as code_write if target filename has a source-code extension,
    # otherwise it'll be tagged file_write via the redirect pattern above.
    ("code_write", re.compile(
        r"cat\s*(<<-?\s*['\"]?\w+['\"]?)?\s*>\s*['\"]?[\w./\-]+\.(py|pyi|js|mjs|ts|tsx|jsx|c|cc|cpp|h|hpp|java|go|rs|rb|pl|php|sh|bash|zsh|lua|r|sql|yml|yaml|toml|json|md|html|css|xml)['\"]?",
        re.IGNORECASE,
    )),
    ("code_write", re.compile(
        r"(echo|printf)\s+.*>\s*['\"]?[\w./\-]+\.(py|pyi|js|mjs|ts|tsx|jsx|c|cc|cpp|h|hpp|java|go|rs|rb|pl|php|sh|bash|zsh|lua|r|sql)['\"]?",
        re.IGNORECASE,
    )),

    # Running tests / verifiers
    ("verify", re.compile(r"\bpython3?\s+-m\s+(pytest|unittest)\b")),
    ("verify", re.compile(r"\[\s+-[fderLswxn]\s+")),  # [ -f path ], [ -d path ], etc.
    ("verify", re.compile(r"\btest\s+-[fderLswxn]\b")),

    # Service start/stop
    ("service", re.compile(r"\b(systemctl|service)\s+(start|stop|restart|status|enable|reload)\b")),

    # git usage is often file manipulation-esque; we tag it separately only if needed.
]


# ---------------------------------------------------------------------------
# Splitting helpers
# ---------------------------------------------------------------------------

# Note: this splitter is *approximate* — it doesn't parse POSIX shell.  It's
# good enough for categorising the short ad-hoc commands our agents emit.
_SPLIT_RE = re.compile(r"\s*(?:\|\||&&|;|\n|\|)\s*")


def _split_subcommands(command: str) -> List[str]:
    """Split a one-liner into individual simple commands (approximate)."""
    # Protect heredoc bodies so we don't split them mid-pipeline.
    heredocs: List[str] = []

    def _stash_heredoc(m: re.Match) -> str:
        heredocs.append(m.group(0))
        return f"__HEREDOC_{len(heredocs) - 1}__"

    hd_re = re.compile(
        r"<<-?\s*['\"]?(\w+)['\"]?.*?\n\1",
        re.DOTALL,
    )
    stashed = hd_re.sub(_stash_heredoc, command)
    parts = [p.strip() for p in _SPLIT_RE.split(stashed) if p.strip()]
    # Re-inject heredoc markers so downstream patterns can still find them.
    for i, h in enumerate(heredocs):
        parts = [p.replace(f"__HEREDOC_{i}__", h) for p in parts]
    return parts


_HEAD_RE = re.compile(r"^\s*(?:sudo\s+(?:-\S+\s+)*)?([\w./-]+)")


def _head_token(subcmd: str) -> str:
    """Return the first token of a command (argv[0]), handling `sudo` prefix."""
    m = _HEAD_RE.match(subcmd)
    if not m:
        return ""
    tok = m.group(1)
    # Strip leading path components: /usr/bin/python3 -> python3
    return tok.rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_one(command: str) -> Set[str]:
    """Return the set of category tags for a single bash one-liner."""
    if not command or not command.strip():
        return set()
    tags: Set[str] = set()
    subcmds = _split_subcommands(command)
    if not subcmds:
        subcmds = [command]
    for sub in subcmds:
        head = _head_token(sub)
        matched_here: Set[str] = set()
        for cat, heads in _HEAD_TOKENS.items():
            if head in heads:
                matched_here.add(cat)
        for cat, pat in _SUBSTRING_PATTERNS:
            if pat.search(sub):
                matched_here.add(cat)

        # Refinement: if we saw both `code_write` and a plain `file_write`,
        # keep only `code_write` (more specific).
        if "code_write" in matched_here and "file_write" in matched_here:
            matched_here.discard("file_write")

        # Refinement: `cat > foo.py` matches both `file_read` (head=cat) and
        # `code_write` (regex); the intent is writing, so drop the read tag.
        if ("code_write" in matched_here or "file_write" in matched_here) \
                and "file_read" in matched_here:
            matched_here.discard("file_read")

        # Refinement: `pkg_install` is more informative than the bare
        # `pip`/`apt` head-token hit it would otherwise get via `code_run`.
        if "pkg_install" in matched_here:
            matched_here.discard("code_run")

        if not matched_here:
            matched_here.add("other")
        tags |= matched_here
    return tags


def tag_histogram(commands: Iterable[str]) -> Counter:
    """Aggregate tags across many commands — each command contributes its tags once."""
    c: Counter = Counter()
    for cmd in commands:
        for tag in classify_one(cmd):
            c[tag] += 1
    return c


# ---------------------------------------------------------------------------
# CLI for quick inspection
# ---------------------------------------------------------------------------


def _cli() -> None:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Classify bash commands into our taxonomy.")
    ap.add_argument(
        "--stdin",
        action="store_true",
        help="Read one command per line from stdin and print 'cmd\\ttag1,tag2'",
    )
    ap.add_argument("commands", nargs="*", help="Commands to classify")
    args = ap.parse_args()

    cmds = args.commands
    if args.stdin:
        cmds = cmds + [line.rstrip() for line in sys.stdin if line.rstrip()]

    for c in cmds:
        tags = sorted(classify_one(c))
        print(f"{','.join(tags)}\t{c}")


if __name__ == "__main__":
    _cli()
