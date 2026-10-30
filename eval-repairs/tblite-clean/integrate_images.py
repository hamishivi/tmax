"""Make the three dependency repair recipes self-contained Harbor contexts."""

import argparse
import re
from pathlib import Path


def compose(original, repair, base):
    lines = original.splitlines()
    starts = [i for i, line in enumerate(lines) if re.match(r"(?i)^FROM\s", line)]
    if len(starts) != 1 or re.search(r"(?i)\sAS\s", lines[starts[0]]):
        raise ValueError("Expected the reviewed single-stage upstream Dockerfile")
    expected = "FROM " + base
    if repair.count(expected) != 1:
        raise ValueError("Repair must name its expected upstream base exactly once")
    lines[starts[0]] += " AS tblite_upstream"
    return "\n".join(lines) + "\n\n" + repair.replace(expected, "FROM tblite_upstream", 1)


def integrate(tasks, recipes):
    for name, recipe, base in (
        ("maven-slf4j-conflict", "Maven.Dockerfile", "tblite-repair-maven:20260909"),
        ("okhttp-trailers-crash", "OkHttp.Dockerfile", "tblite-repair-okhttp:20260909"),
        ("breast-cancer-mlflow", "MLflow.Dockerfile", "tblite-repair-mlflow:20260909"),
    ):
        environment = tasks / name / "environment"
        dockerfile = environment / "Dockerfile"
        combined = compose(dockerfile.read_text(), (recipes / recipe).read_text(), base)
        if name == "maven-slf4j-conflict":
            solution = (tasks / name / "solution/solve.sh").read_text()
            marker = "cat > pom.xml << 'EOF'\n"
            if solution.count(marker) != 1:
                raise ValueError("Expected the reviewed Maven dependency POM heredoc")
            body = solution.split(marker, 1)[1]
            if "\nEOF" not in body:
                raise ValueError("Unterminated dependency POM")
            with (environment / "dependency-pom.xml").open("x") as target:
                target.write(body.split("\nEOF", 1)[0])
        elif name == "okhttp-trailers-crash":
            with (environment / "okhttp-cache.gradle").open("x") as target:
                target.write((recipes / "okhttp-cache.gradle").read_text())
        dockerfile.write_text(combined)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--recipes", required=True, type=Path)
    args = parser.parse_args()
    integrate(args.tasks, args.recipes)


if __name__ == "__main__":
    main()
