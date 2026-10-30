"""Package React's offline dependencies without changing its grader."""

import argparse
import json
import re
import shlex
import shutil
from pathlib import Path


def prepare(source, output, cache_packages=()):
    for package in cache_packages:
        if not re.fullmatch(r"(?:@[a-z0-9._-]+/)?[a-z0-9._-]+@\d+\.\d+\.\d+(?:-[a-zA-Z0-9.-]+)?", package):
            raise ValueError("Extra cache entries must be registry package@exact-version")
    solution = (source / "solution/solve.sh").read_text()
    marker = "cat > package.json << 'EOF'\n"
    if solution.count(marker) != 1:
        raise ValueError("Expected the pinned dependency manifest")
    package = json.loads(solution.split(marker, 1)[1].split("\nEOF", 1)[0])
    dependencies = {key: package[key] for key in ("dependencies", "devDependencies")}
    dockerfile = (source / "environment/Dockerfile").read_text()
    if dockerfile.count("FROM ") != 1:
        raise ValueError("Expected the pinned single-stage image")
    lines = dockerfile.splitlines()
    lines[0] += " AS original"
    dockerfile = (
        "\n".join(lines)
        + """

RUN npm install -g --prefix /usr/local node@20.19.6 npm@10.9.2 \
    && hash -r && node --version && npm --version \
    && test "$(node --version)" = v20.19.6 \
    && test "$(npm --version)" = 10.9.2
RUN pip install --no-cache-dir uv==0.9.5
ENV UV_CACHE_DIR=/opt/uv-cache UV_LINK_MODE=copy
RUN uvx -p 3.13 -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 pytest --version

FROM original AS dependency_cache
WORKDIR /tmp/dependencies
COPY dependency-package.json package.json
RUN npm install --ignore-scripts --audit=false --fund=false

FROM original
COPY --from=dependency_cache /root/.npm /opt/npm-cache
ENV npm_config_cache=/opt/npm-cache npm_config_offline=true
ENV npm_config_audit=false npm_config_fund=false UV_OFFLINE=1
"""
    )
    extra_cache = "".join(
        f"RUN mkdir /tmp/cache-extra-{index} && cd /tmp/cache-extra-{index} "
        "&& npm install --ignore-scripts --audit=false --fund=false -- "
        f"{shlex.quote(package)}\n"
        for index, package in enumerate(cache_packages)
    )
    dockerfile = dockerfile.replace(
        "\nFROM original\nCOPY --from=dependency_cache",
        f"\n{extra_cache}\nFROM original\nCOPY --from=dependency_cache",
    )
    verifier = (source / "tests/test.sh").read_text()
    start = "# Install system dependencies\n"
    end = "# Check if we're in a valid working directory\n"
    if any(verifier.count(marker) != 1 for marker in (start, end)):
        raise ValueError("Unexpected React verifier bootstrap")
    verifier = (
        verifier[: verifier.index(start)]
        + """# Dependencies were provisioned in the image; retain the original grader.
node --version
npm --version

"""
        + verifier[verifier.index(end) :]
    )
    shutil.copytree(source, output)
    (output / "environment/dependency-package.json").write_text(json.dumps(dependencies, indent=2) + "\n")
    (output / "environment/Dockerfile").write_text(dockerfile)
    (output / "tests/test.sh").write_text(verifier)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cache-package",
        action="append",
        default=[],
        help="Additional npm package@version to cache with its dependencies; repeatable",
    )
    args = parser.parse_args()
    prepare(args.source, args.output, args.cache_package)
