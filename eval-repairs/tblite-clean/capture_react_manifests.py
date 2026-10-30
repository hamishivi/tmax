"""Runner-side capture of final agent manifests, before verifier mutation."""

import hashlib
import json
import shlex
from pathlib import Path


async def capture(environment, output, workdir="/app"):
    """Call after agent execution and before verification on the same sandbox.

    This only copies fixed manifest names; it does not run npm or change scores.
    Let the runner report capture failures separately from the verifier result.
    """
    output = Path(output)
    if not workdir.startswith("/"):
        raise ValueError("workdir must be absolute")
    output.mkdir(parents=True, exist_ok=False)
    inventory = {}
    for name in ("package.json", "package-lock.json", "npm-shrinkwrap.json"):
        source = f"{workdir.rstrip('/')}/{name}"
        result = await environment.exec(f"test -f {shlex.quote(source)}", timeout_sec=10)
        if result.exit_code == 1:
            inventory[name] = None
            continue
        if result.exit_code != 0:
            raise RuntimeError(f"Cannot inspect {source}: exit {result.exit_code}")
        target = output / name
        await environment.download_file(source, target)
        inventory[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    (output / "manifest-sha256.json").write_text(json.dumps(inventory, indent=2) + "\n")
    return inventory
