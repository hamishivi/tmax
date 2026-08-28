"""Shared sandbox backend contract and lifecycle errors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result from code or command execution."""

    stdout: str
    stderr: str
    exit_code: int


class SandboxOOMError(RuntimeError):
    """Raised when the sandbox container was killed by the OOM reaper.

    Callers should treat this as terminal for the current episode rather than
    retrying because the next command will almost certainly hit the same limit.
    """


class SandboxLostError(RuntimeError):
    """Raised when infrastructure hosting a sandbox disappears.

    The current episode cannot resume because its ephemeral filesystem is
    gone, but the environment actor may be reused after reset.
    """


class SandboxBackend(ABC):
    """Abstract interface for code and command execution backends."""

    @abstractmethod
    def start(self) -> None:
        """Initialize the sandbox before other operations."""

    def restart(self) -> None:
        """Replace the sandbox while retaining reusable backend resources.

        Backends without a cheaper reset path use the default close/start
        behavior. Backends with an outer isolation boundary may override it.
        """
        self.close()
        self.start()

    @abstractmethod
    def run_command(self, command: str, timeout: int | None = None) -> ExecutionResult:
        """Execute a shell command in the sandbox."""

    @abstractmethod
    def write_file(self, path: str, content: str | bytes) -> None:
        """Write a file to the sandbox filesystem."""

    @abstractmethod
    def read_file(self, path: str, binary: bool = False) -> str | bytes:
        """Read a file from the sandbox filesystem."""

    @abstractmethod
    def put_archive(self, root: str, tar_bytes: bytes) -> None:
        """Extract a tar archive inside the sandbox, rooted at ``root``."""

    @abstractmethod
    def close(self) -> None:
        """Clean up sandbox resources."""
